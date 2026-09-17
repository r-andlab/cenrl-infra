"""
ClusterLinUCB — HATCH-style semantic cluster oracle, fixing HierLinUCB's
inert propagation mechanisms.

Context
-------
HierLinUCB (models/contextual_bandit/hier_linucb.py) has two propagation
mechanisms that turned out weaker than intended:
  1. Upward path propagation is a structural no-op under the canonical
     FEATURES=["categories"] config — every category's parent is root, so
     there's no ancestor level to propagate to.
  2. Semantic sibling propagation only updates b (not A) for sibling arms,
     so a "transferred" arm's predicted mean moves but its exploration
     bonus stays exactly as high as an untouched arm's — the confidence
     estimate never reflects the borrowed evidence.
Widening sibling-group coverage (HierLinUCB v2, hier_linucb_v2.py) made
results WORSE, not better (China 63.2%->57.7%, Kazakhstan 63.9%->54.8%):
loose groups (e.g. lumping 20 unrelated "Lifestyle" categories together) let
one positive probe inflate predictions for categories with no real
correlation.

Fix tested here
----------------
This model changes the propagation MECHANISM, not the grouping — it reuses
the exact same six semantic groups the original HierLinUCB used
(CIRCUMVENTION_CATS, ADULT_CATS, NEWS_CATS, SOCIAL_CATS, SEARCH_CATS,
STREAMING_CATS, all defined in linucb.py), which we already know don't hurt.
Instead of a b-only gamma-weighted nudge to individual siblings, each group
gets a real cluster-level oracle (A_cluster/b_cluster) updated with a FULL
outer-product update every time any category in that group is probed —
exactly HATCH's cluster/arm split (models/contextual_bandit/hatch.py),
except the cluster level here is a semantically coherent category group
instead of HATCH's TLD level. TLD carries little signal in this action space
(~90% of arms are .com), which is likely why HATCH underperformed in
testing; category groups carry real signal, which is why the original
HierLinUCB helped at all on Russia/Kazakhstan despite its weak mechanism.

This isolates the experiment to "does a properly-tracked cluster oracle
(mean AND uncertainty both shrink with pooled evidence) beat a bare b-nudge",
holding the grouping choice fixed at the one already validated as safe.

Score(a) = category_ucb(a) + cluster_ucb(group(a))     [group(a) may be None]

  category_ucb(a) = theta_a . x_a + alpha_arm * sqrt(x_a . A_a^-1 . x_a)
  cluster_ucb(g)  = theta_g . x_a + alpha_cluster * sqrt(x_a . A_g^-1 . x_a)

Categories outside all six groups fall back to category_ucb alone —
identical to plain disjoint LinUCB for those arms, same as in HierLinUCB.

Update: probing category a (in group g) does a FULL update to both A_a/b_a
(as in disjoint LinUCB) AND A_g/b_g (as in HATCH's cluster oracle) — not the
b-only nudge HierLinUCB used.

Usage
-----
    python3 models/contextual_bandit/cluster_linucb.py \\
          -m 2000 -E 10 -alpha 0.5 \\
          -o outputs/cluster_linucb \\
          -g inputs/gfwatch/gfwatch-blocklist.csv \\
          -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \\
          -f categories
"""

import typing

import numpy as np

from models.base.action_space import ACTION_ATTEMPTS, Q_VALUE, NAME
from models.base.model import run_multiprocessing
from models.contextual_bandit.linucb import (
    LinUCB, LinUCBActionSpace, LinUCBParserOptions, FEATURE_DIM,
    CIRCUMVENTION_CATS, ADULT_CATS, NEWS_CATS, SOCIAL_CATS, SEARCH_CATS, STREAMING_CATS,
)

np.seterr(divide="ignore", invalid="ignore")

# Same six groups as the original HierLinUCB (see linucb.py) — deliberately
# NOT expanded, since widening coverage (HierLinUCB v2) made results worse.
_CLUSTER_GROUPS: typing.Dict[str, typing.Set[str]] = {
    "circumvention": CIRCUMVENTION_CATS,
    "adult":         ADULT_CATS,
    "news":          NEWS_CATS,
    "social":        SOCIAL_CATS,
    "search":        SEARCH_CATS,
    "streaming":     STREAMING_CATS,
}


def _cluster_of(node_name: str) -> typing.Optional[str]:
    for cluster, cats in _CLUSTER_GROUPS.items():
        if node_name in cats:
            return cluster
    return None


class ClusterLinUCB(LinUCB):
    """
    LinUCB with a HATCH-style semantic cluster oracle layered on top of the
    existing per-category oracle (see module docstring). Inherits feature
    construction and greedy arm-selection traversal unchanged from LinUCB;
    only scoring and the update rule differ.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.alpha_arm     = params.get("alpha_arm", self.alpha)
        self.alpha_cluster = params.get("alpha_cluster", self.alpha)

        self._A_cluster: typing.Dict[str, np.ndarray] = {}
        self._b_cluster: typing.Dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self._A_cluster.clear()
        self._b_cluster.clear()

    def _ensure_cluster(self, cluster: str):
        if cluster not in self._A_cluster:
            self._A_cluster[cluster] = np.eye(FEATURE_DIM)
            self._b_cluster[cluster] = np.zeros(FEATURE_DIM)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _linucb_score(self, node_key: str) -> float:
        self._ensure_matrices(node_key)
        x = self._get_feature_vector(node_key)
        node_name = self.action_space.get(node_key)[NAME]

        A_arm, b_arm = self._A[node_key], self._b[node_key]
        theta    = np.linalg.solve(A_arm, b_arm)
        arm_pred = float(theta @ x)
        arm_unc  = float(np.sqrt(x @ np.linalg.solve(A_arm, x)))
        score = arm_pred + self.alpha_arm * arm_unc

        cluster = _cluster_of(node_name)
        if cluster is not None:
            self._ensure_cluster(cluster)
            A_c, b_c = self._A_cluster[cluster], self._b_cluster[cluster]
            theta_c   = np.linalg.solve(A_c, b_c)
            clus_pred = float(theta_c @ x)
            clus_unc  = float(np.sqrt(x @ np.linalg.solve(A_c, x)))
            score += clus_pred + self.alpha_cluster * clus_unc

        return score

    # ── Update ────────────────────────────────────────────────────────────────

    def observe(self, selected_arm: str, measurement_result: float) -> float:
        self._ensure_matrices(selected_arm)
        x = self._get_feature_vector(selected_arm)
        node_name = self.action_space.get(selected_arm)[NAME]

        # Full update to the category's own arm oracle (same as disjoint LinUCB)
        self._A[selected_arm] += np.outer(x, x)
        self._b[selected_arm] += measurement_result * x

        # Full update to the cluster oracle, if this category belongs to one
        cluster = _cluster_of(node_name)
        if cluster is not None:
            self._ensure_cluster(cluster)
            self._A_cluster[cluster] += np.outer(x, x)
            self._b_cluster[cluster] += measurement_result * x

        theta = np.linalg.solve(self._A[selected_arm], self._b[selected_arm])
        predicted = float(theta @ x)
        if cluster is not None:
            theta_c = np.linalg.solve(self._A_cluster[cluster], self._b_cluster[cluster])
            predicted += float(theta_c @ x)
        predicted = float(np.clip(predicted, 0.0, 1.0))

        n_data = self.action_space.get(selected_arm)
        n_data[ACTION_ATTEMPTS] += 1
        n_data[Q_VALUE] = round(predicted, 2)
        return n_data[Q_VALUE]


class ClusterLinUCBParserOptions(LinUCBParserOptions):
    def add_arguments(self):
        super().add_arguments()
        self.parser.add_argument(
            "--alpha_arm", type=float, default=None,
            help="Exploration bonus for the per-category oracle (default: same as alpha)",
        )
        self.parser.add_argument(
            "--alpha_cluster", type=float, default=None,
            help="Exploration bonus for the semantic cluster oracle (default: same as alpha)",
        )

    def set_params(self, args):
        super().set_params(args)
        self.params["alpha_arm"]     = args.alpha_arm     if args.alpha_arm     is not None else args.alpha
        self.params["alpha_cluster"] = args.alpha_cluster if args.alpha_cluster is not None else args.alpha


if __name__ == "__main__":
    parser = ClusterLinUCBParserOptions()
    params = parser.parse()
    run_multiprocessing(ClusterLinUCB, params,
                        addition_model_run_kwargs={
                            "action_space_klass": LinUCBActionSpace,
                        })
