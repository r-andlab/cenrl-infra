"""
Contextual Bandit

check equaqtion
    score(a) = θ_a · x_a  +  α * sqrt(x_a · A_a⁻¹ · x_a)
               ─────────────  ───────────────────────────────
               predicted       uncertainty bonus
               reward


  x_a  = feature vector describing arm a
  A_a  = I + Σ x_t x_t^T  (updated after each probe in this arm)
  b_a  = Σ r_t x_t         (reward-weighted feature accumulator)
  θ_a  = A_a⁻¹ b_a         (learned weight vector)
  α    = exploration parameter (tunable)



Reference:
  Li et al., "A Contextual-Bandit Approach to Personalized News Article
  Recommendation", WWW 2010.


Usage:
    python3 models/contextual_bandit/linucb.py \
        -m 1000 -E 3 -alpha 0.5 \
        -o outputs/linucb \
        -g inputs/gfwatch/gfwatch-blocklist.csv \
        -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \
        -f categories
"""

import os
import typing
from random import randrange
from urllib.parse import urlparse

import numpy as np
import pandas as pd

import models.base.action_space as action_space_module
from models.base.action_space import (
    ActionSpaceBase, NODE_TYPE_KEY, POSSIBLE_TARGET_FEATURES,
    ACTION_ATTEMPTS, Q_VALUE, NAME, IS_TARGET_NODE
)
from models.base.model import Model, ParserOptions, run_multiprocessing

np.seterr(divide="ignore", invalid="ignore")

# Feature groupings (consistent w/ analyze_features.py) 

CIRCUMVENTION_CATS = {"P2P", "Anonymizer", "File Sharing", "Redirect", "Hacking"}
ADULT_CATS         = {"Pornography", "Adult Themes", "Dating & Relationships",
                      "Nudity", "Lingerie & Bikini", "Sex Education"}
NEWS_CATS          = {"News & Media", "Magazines",
                      "Politics, Advocacy, and Government-Related",
                      "Forums", "Personal Blogs", "News, Portal & Search"}
SOCIAL_CATS        = {"Social Networks", "Instant Messengers", "Chat",
                      "Professional Networking", "Messaging"}
SEARCH_CATS        = {"Search Engines", "News, Portal & Search"}
STREAMING_CATS     = {"Video Streaming", "Audio Streaming", "Television",
                      "Music", "Radio"}

# TLD flags
CN_TLD  = {"cn"}                    # Chinese ccTLD — high block rate
COM_TLD = {"com"}                  
ORG_TLD = {"org"}              
NET_TLD = {"net"}                   
# is_cctld: True when node_name is a 2-letter ISO country code//////

# Feature vector dimension  _static_features() + _get_feature_vector()
# dims 0-5:  semantic category flags
# dims 6-10: TLD flags (cn, com, org, net, any-cctld)
# dim  11:   log-normalised domain count
# dim  12:   fraction of domains not yet probed (dynamic)
FEATURE_DIM = 13


def _static_features(node_name: str) -> np.ndarray:
    """
    11-dimensional static feature vector derived from the arm's name alone.
    dims 0-5:  semantic content-category flags
    dims 6-10: TLD-family flags (cn / com / org / net / any-ccTLD)
    """
    cats = {node_name}
    tld  = node_name.lower()
    return np.array([
        # content-category flags
        float(bool(cats & CIRCUMVENTION_CATS)),   # 0
        float(bool(cats & ADULT_CATS)),            # 1
        float(bool(cats & NEWS_CATS)),             # 2
        float(bool(cats & SOCIAL_CATS)),           # 3
        float(bool(cats & SEARCH_CATS)),           # 4
        float(bool(cats & STREAMING_CATS)),        # 5
        # TLD-family flags
        float(tld in CN_TLD),                      # 6  .cn
        float(tld in COM_TLD),                     # 7  .com
        float(tld in ORG_TLD),                     # 8  .org
        float(tld in NET_TLD),                     # 9  .net
        float(len(tld) == 2),                      # 10 any ccTLD (2-letter)
    ], dtype=float)


class LinUCBActionSpace(ActionSpaceBase):
    """
    Extends ActionSpaceBase to pre-compute the total domain count per arm.
     count used for the fraction_remaining dynamic feature.
    """
    def build_graph(self):
        g = super().build_graph()
        # for each non-target node, count how many domain leaves sit below it
        root = action_space_module.ROOT_KEY
        for node, n_data in g.nodes(data=True):
            if node == root or n_data.get(IS_TARGET_NODE, False):
                continue
            total = sum(
                1 for d in g.successors(node)
                if g.nodes[d].get(IS_TARGET_NODE, False)
            )
            n_data["total_domains"] = max(total, 1)   # avoid div-by-zero

        return g


class LinUCB(Model):
    """
    bandit:  Ttaverses the hierarchy like UCB but
    replaces Q + exploration_bonus score with the LinUCB formula.
     arm maintains its own A matrix and b vector; the weight vector θ
     solved lazily at scoring time via np.linalg.solve.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.alpha = params["alpha"]
        self.initial_value_estimate = params["initial_value_estimate"]
        # per-arm LinUCB state: node_key -> np.ndarray
        # ppulated lazily on first call to choose_arm()
        self._A: typing.Dict[str, np.ndarray] = {}   # FEATURE_DIM × FEATURE_DIM
        self._b: typing.Dict[str, np.ndarray] = {}   # FEATURE_DIM

        self.exploration_epoch_num = self.current_epoch_num

    #  Initialisatiom

    def _ensure_matrices(self, node_key: str):
        """Lazily initialise A and b for a node the first time it is seen."""
        if node_key not in self._A:
            self._A[node_key] = np.eye(FEATURE_DIM)
            self._b[node_key] = np.zeros(FEATURE_DIM)

    def reset(self):
        super().reset()
        self._A.clear()
        self._b.clear()
        self.exploration_epoch_num = self.current_epoch_num

    #  Feature construction 

    def _get_feature_vector(self, node_key: str) -> np.ndarray:
        """
        13-dimensional feature vector for an arm node.

        Dims  0-5:  static semantic category flags
        Dims  6-10: static TLD-family flags (cn, com, org, net, any-ccTLD)
        Dim   11:   log-normalised total domain count of this arm
        Dim   12:   fraction of domains not yet probed (dynamic)
        """
        n_data = self.action_space.get(node_key)
        node_name = n_data[NAME]

        static = _static_features(node_name)

        # total_domains was stored by LinUCBActionSpace.build_graph()
        total = n_data.get("total_domains", 1)
        log_size = np.log(total + 1) / np.log(10_001)   # normalised 0-1

        # count (unprobed) direct domain children
        graph = self.action_space.get_graph()
        active = sum(
            1 for c in graph.successors(node_key)
            if graph.nodes[c].get(IS_TARGET_NODE, False)
               and not graph.nodes[c].get(action_space_module.SLEEPING, False)
        )
        fraction_remaining = active / total

        return np.concatenate([static, [log_size, fraction_remaining]])

    #  LinUCB scoring 

    def _linucb_score(self, node_key: str) -> float:
        """compute upper confidence bound for one arm."""
        self._ensure_matrices(node_key)
        x   = self._get_feature_vector(node_key)
        A   = self._A[node_key]
        b   = self._b[node_key]

        # solve A θ = b  (more numerically stable than explicit inversion)
        theta = np.linalg.solve(A, b)

        predicted    = float(theta @ x)
        A_inv_x      = np.linalg.solve(A, x)
        uncertainty  = float(np.sqrt(x @ A_inv_x))

        return predicted + self.alpha * uncertainty

    # Arm selection

    def choose_arm(self) -> typing.List[str]:
        """
        Traverse the hierarchy layer by layer.
        At each layer, score every child arm with LinUCB and pick the argmax.
        identical to UCBNaive.choose_arm()
        """
        source = self.action_space.get_root()
        selected_arm = None
        reached_target_nodes = False
        selected_arms_history = []

        while not reached_target_nodes:
            immediate_children = []
            scores = []

            for succ in self.action_space.get_graph().successors(source):
                succ_data = self.action_space.get(succ)
                if succ_data[NODE_TYPE_KEY] in POSSIBLE_TARGET_FEATURES:
                    reached_target_nodes = True
                    break

                immediate_children.append(succ)
                scores.append(self._linucb_score(succ))

            if reached_target_nodes:
                break

            if immediate_children:
                scores_arr = np.array(scores)
                # break ties randomly (same as UCB)
                top_indices = np.argwhere(scores_arr == scores_arr.max()).flatten()
                selected_arm = immediate_children[top_indices[randrange(len(top_indices))]]
                selected_arms_history.append(selected_arm)

            source = selected_arm

        # increment attempts for all arms except last (updated in observe())
        for a in selected_arms_history[:-1]:
            self.action_space.get(a)[ACTION_ATTEMPTS] += 1

        self.last_selected_arm_index = selected_arms_history[-1]
        return selected_arms_history

    # Weight update

    def observe(self, selected_arm: str, measurement_result: float) -> float:
        """
        Update A and b for the selected arm using  observed reward,
        then solve for θ and store the predicted val as Q_VALUE,
        propagate_rewards() + the CSV output work unchanged.
        """
        self._ensure_matrices(selected_arm)
        x = self._get_feature_vector(selected_arm)

        # LinUCB update
        self._A[selected_arm] += np.outer(x, x)
        self._b[selected_arm] += measurement_result * x

        # derive predicted reward from updated weights
        theta = np.linalg.solve(self._A[selected_arm], self._b[selected_arm])
        predicted = float(np.clip(theta @ x, 0.0, 1.0))

        n_data = self.action_space.get(selected_arm)
        n_data[ACTION_ATTEMPTS] += 1
        n_data[Q_VALUE] = round(predicted, 2)

        return n_data[Q_VALUE]

    def step(self) -> dict:
        self.exploration_epoch_num += 1
        return super().step()


# Blocklist feature helpers

def _load_reference_domains(lists_dir: str) -> typing.Set[str]:
    """
    Load all Citizen Lab country-test-list CSVs from lists_dir and return
    the set of bare domains extracted from their URL entries.
    """
    domains: typing.Set[str] = set()
    for fname in os.listdir(lists_dir):
        if not fname.endswith(".csv"):
            continue
        try:
            df = pd.read_csv(os.path.join(lists_dir, fname))
            if "url" in df.columns:
                for raw in df["url"].dropna():
                    host = urlparse(str(raw)).netloc.lower().strip()
                    if host.startswith("www."):
                        host = host[4:]
                    if host:
                        domains.add(host)
            elif "domain" in df.columns:
                for d in df["domain"].dropna():
                    domains.add(str(d).lower().strip())
        except Exception:
            continue
    return domains


FEATURE_DIM_BLOCKLIST = 14  # 13 base features + 1 blocklist-coverage feature


class LinUCBBlocklistActionSpace(LinUCBActionSpace):
    """
    extends LinUCBActionSpace for each arm, the fraction of
    its domain children that appear in the Citizen Lab reference lists.
    """
    reference_lists_dir: typing.ClassVar[typing.Optional[str]] = None
    _cached_ref_domains: typing.ClassVar[typing.Optional[typing.Set[str]]] = None

    def build_graph(self):
        if (LinUCBBlocklistActionSpace._cached_ref_domains is None
                and LinUCBBlocklistActionSpace.reference_lists_dir):
            print("LinUCBBlocklistActionSpace: loading reference blocklists …")
            LinUCBBlocklistActionSpace._cached_ref_domains = _load_reference_domains(
                LinUCBBlocklistActionSpace.reference_lists_dir
            )
            print(f"  loaded {len(LinUCBBlocklistActionSpace._cached_ref_domains):,} reference domains")

        ref = LinUCBBlocklistActionSpace._cached_ref_domains or set()
        g = super().build_graph()

        root = action_space_module.ROOT_KEY
        for node, n_data in g.nodes(data=True):
            if node == root or n_data.get(IS_TARGET_NODE, False):
                continue
            total = n_data.get("total_domains", 1)
            in_ref = sum(
                1 for d in g.successors(node)
                if g.nodes[d].get(IS_TARGET_NODE, False)
                   and g.nodes[d].get(NAME, "").lower() in ref
            )
            n_data["blocklist_frac"] = in_ref / total

        return g


class LinUCBBlocklist(LinUCB):
    """
    LinUCB with an extra 14th feature: fraction of the arm's domains that
    appear in the Citizen Lab reference country lists.
    """

    def _ensure_matrices(self, node_key: str):
        if node_key not in self._A:
            self._A[node_key] = np.eye(FEATURE_DIM_BLOCKLIST)
            self._b[node_key] = np.zeros(FEATURE_DIM_BLOCKLIST)

    def _get_feature_vector(self, node_key: str) -> np.ndarray:
        base = super()._get_feature_vector(node_key)
        n_data = self.action_space.get(node_key)
        blocklist_frac = float(n_data.get("blocklist_frac", 0.0))
        return np.append(base, blocklist_frac)


# Parser/entry

class LinUCBParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument(
            "-alpha", "--alpha", type=float, default=0.5,
            help="LinUCB exploration parameter (higher = more exploration)"
        )
        self.parser.add_argument(
            "-V", "--initialvalueestimate", type=float, default=0.0
        )
        self.parser.add_argument(
            "-X", "--actionvaluefile", type=str, default=None
        )

    def set_params(self, args):
        super().set_params(args)
        self.params["alpha"] = args.alpha
        self.params["initial_value_estimate"] = args.initialvalueestimate
        self.params["action_value_file"] = args.actionvaluefile


if __name__ == "__main__":
    parser = LinUCBParserOptions()
    params = parser.parse()
    run_multiprocessing(LinUCB, params,
                        addition_model_run_kwargs={
                            "action_space_klass": LinUCBActionSpace
                        })
