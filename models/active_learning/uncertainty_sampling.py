"""
Active learning via uncertainty sampling 

Usage (same interface as other models):
    python3 models/active_learning/uncertainty_sampling.py \
        -m 1000 -E 3 \
        -o outputs/uncertainty_sampling \
        -g inputs/gfwatch/gfwatch-blocklist.csv \
        -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \
        -f categories
"""

import random
import typing
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

import models.base.action_space as action_space_module
from models.base.model import Model, ParserOptions, run_multiprocessing

#  Feature groupings 

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
MAJOR_US_TECH      = {"Google LLC", "Meta Platforms, Inc.", "Twitter, Inc.",
                      "Amazon.com, Inc.", "Microsoft Corporation",
                      "Apple Inc.", "Alphabet Inc."}

# Min labeled examples needed before the classifier is used.
# Below ,selection is random (cold-start phase).
MIN_SAMPLES_TO_FIT = 20


def _build_feature_vector(categories: typing.List[str],
                           entity: str,
                           rank: int) -> np.ndarray:
    """
    10-dimensional feature vector for a single domain.
    All features are either binary (0/1) or a normalized continuous value.
    """
    cats = set(categories)
    return np.array([
        float(bool(cats & CIRCUMVENTION_CATS)),   # 0
        float(bool(cats & ADULT_CATS)),            # 1
        float(bool(cats & NEWS_CATS)),             # 2
        float(bool(cats & SOCIAL_CATS)),           # 3
        float(bool(cats & SEARCH_CATS)),           # 4
        float(bool(cats & STREAMING_CATS)),        # 5
        float(str(entity) in MAJOR_US_TECH),       # 6
        float(rank <= 200),                        # 7  top-200 flag
        float(5000 <= rank <= 7000),               # 8  mid-tier flag
        np.log(rank + 1) / np.log(10_001),         # 9  normalized log rank
    ], dtype=float)


class UncertaintySampling(Model):


    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

        # domain_name -> 10-d feature vector, built from the Tranco CSV
        self._feature_map: typing.Dict[str, np.ndarray] = {}

        # domain_name -> observed label (0 = not blocked, 1 = blocked)
        self._labels: typing.Dict[str, int] = {}

        # sklearn classifier — re-fitted after every probe once we have enough data
        self._clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        self._clf_fitted = False

        self._build_feature_map()

    #  Initialisation 

    def _build_feature_map(self):
        """Read Tranco CSV and build a feature vector for every domain."""
        df = pd.read_csv(self.action_space_file, delimiter="|", index_col=False)
        df["categories"] = df["categories"].apply(literal_eval)

        for _, row in df.iterrows():
            domain  = row["domain"]
            cats    = row["categories"] if isinstance(row["categories"], list) else []
            entity  = row["entity"] if pd.notna(row.get("entity")) else ""
            rank    = int(row["rank"])
            self._feature_map[domain] = _build_feature_vector(cats, entity, rank)

    def reset(self):
        """Called between episodes: reset labels and classifier."""
        super().reset()
        self._labels.clear()
        self._clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        self._clf_fitted = False

    #  Classifier 

    def _fit_classifier(self):
        """Re-fit on all labeled data so far """
        if len(self._labels) < MIN_SAMPLES_TO_FIT:
            return

        domains_with_features = [d for d in self._labels if d in self._feature_map]
        if not domains_with_features:
            return

        X = np.array([self._feature_map[d] for d in domains_with_features])
        y = np.array([self._labels[d]       for d in domains_with_features])

        # need at least one positive (blocked) and one negative to fit
        if len(np.unique(y)) < 2:
            return

        self._clf.fit(X, y)
        self._clf_fitted = True

    #  Selection 

    def _select_domain(self) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
        """
        Return (node_key, domain_name) for  next domain to probe
        """
        # collect all unprobed (active) target nodes
        candidates      = []   # node keys
        candidate_names = []   # domain names

        for node, n_data in self.action_space.gen_active_target_nodes_and_data():
            name = n_data[action_space_module.NAME]
            candidates.append(node)
            candidate_names.append(name)

        if not candidates:
            return None, None

        # cold-start: not enough labeled data yet
        if not self._clf_fitted:
            idx = random.randint(0, len(candidates) - 1)
            return candidates[idx], candidate_names[idx]

        # build feature matrix for all candidates that have features
        feat_indices  = [i for i, n in enumerate(candidate_names) if n in self._feature_map]
        no_feat_indices = [i for i, n in enumerate(candidate_names) if n not in self._feature_map]

        if not feat_indices:
            # no features available -- fall back to random
            idx = random.randint(0, len(candidates) - 1)
            return candidates[idx], candidate_names[idx]

        X_cands = np.array([self._feature_map[candidate_names[i]] for i in feat_indices])
        probs   = self._clf.predict_proba(X_cands)[:, 1]

        # uncertainty = distance from 0.5; smallest = most uncertain
        uncertainty   = np.abs(probs - 0.5)
        best_local    = int(np.argmin(uncertainty))
        best_global   = feat_indices[best_local]

        return candidates[best_global], candidate_names[best_global]

    #  Model interface 

    def choose_arm(self) -> typing.List[str]:
        # Not used -- selection happens globally in step()
        raise NotImplementedError("UncertaintySampling selects targets globally via step()")

    def observe(self, selected_arm: str, measurement_result: float) -> float:
        # No Q-value update needed; return the raw result for logging
        return measurement_result

    def step(self) -> dict:
        """
        Overrides Model.step() to bypass the arm hierarchy entirely.
   
        """
        selected_target, selected_target_name = self._select_domain()

        if selected_target is None:
            # should not happen if can_step() is checked, but guard anyway
            return {
                "action": "none", "target": "none",
                "reward": 0.0, "q_value": 0.0,
                "is_blocked": 0, "is_optimal": 0,
                "coverage": self.get_blocklist_coverage(),
            }

        measurement_result, is_blocked = self.take_measurement(selected_target_name)

        if is_blocked:
            self.update_blocklist_target_found(selected_target_name)

        # label this domain and refit classifier
        self._labels[selected_target_name] = 1 if is_blocked else 0
        self._fit_classifier()

        # remove from action space so it won't be selected again
        self.disable_target(selected_target)

        return {
            "action":     "uncertainty_sampling",
            "target":     selected_target_name,
            "reward":     round(measurement_result, 2),
            "q_value":    round(float(is_blocked), 2),
            "is_blocked": 1 if is_blocked else 0,
            "is_optimal": 0,    # not tracked for this method
            "coverage":   self.get_blocklist_coverage(),
        }


class UncertaintySamplingParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()

    def set_params(self, args):
        super().set_params(args)
        self.params["action_value_file"] = None


if __name__ == "__main__":
    parser = UncertaintySamplingParserOptions()
    params = parser.parse()
    run_multiprocessing(UncertaintySampling, params)
