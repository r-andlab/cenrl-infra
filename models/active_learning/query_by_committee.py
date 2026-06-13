"""
Committee members:
  - Logistic Regression  (linear decision boundary, warm_start for speed)
  - Decision Tree        (non-linear, axis-aligned splits)
  - Naive Bayes          (feature-independent probabilistic model)


  - Committee retrains every REFIT_EVERY probes, not after every single probe
  - Random Forest dropped (too slow per fit); LR+DT+NB gives linear/non-linear/probabilistic diversity
  - LogisticRegression uses warm_start=True to resume from previous weights

Usage:
    python3 models/active_learning/query_by_committee.py \\
        -m 1000 -E 3 \\
        -o outputs/qbc \\
        -g inputs/gfwatch/gfwatch-blocklist.csv \\
        -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv \\
        -f categories
"""

import random
import typing
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import models.base.action_space as action_space_module
from models.base.model import Model, ParserOptions, run_multiprocessing

#  Feature groupings (same as uncertainty_sampling.py) 

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

MIN_SAMPLES_TO_FIT = 20
REFIT_EVERY = 1  # retrain committee every N probes instead of every 1


def _build_feature_vector(categories: typing.List[str],
                           entity: str,
                           rank: int) -> np.ndarray:
    cats = set(categories)
    return np.array([
        float(bool(cats & CIRCUMVENTION_CATS)),
        float(bool(cats & ADULT_CATS)),
        float(bool(cats & NEWS_CATS)),
        float(bool(cats & SOCIAL_CATS)),
        float(bool(cats & SEARCH_CATS)),
        float(bool(cats & STREAMING_CATS)),
        float(str(entity) in MAJOR_US_TECH),
        float(rank <= 200),
        float(5000 <= rank <= 7000),
        np.log(rank + 1) / np.log(10_001),
    ], dtype=float)


def _vote_entropy(probs_matrix: np.ndarray) -> np.ndarray:
    """
    Compute vote entropy across the committee for each candidate domain.

    probs_matrix: shape (n_committee, n_candidates)
        each row is one classifier's predicted block probability for all candidates.

    """
    # Stack into (n_candidates, n_committee) and clip to avoid log(0)
    p = np.clip(probs_matrix.T, 1e-9, 1 - 1e-9)
    q = 1 - p
    # Binary entropy per classifier per candidate, then average across committee
    entropy = -(p * np.log(p) + q * np.log(q))
    return entropy.mean(axis=1)


class QueryByCommittee(Model):
    """
    4 classifiers are trained on all labeled domains after each probe. The next domain to probe is the one with highest average vote entropy across the committee.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

        self._feature_map: typing.Dict[str, np.ndarray] = {}
        self._labels: typing.Dict[str, int] = {}
        self._committee_fitted = False
        self._probes_since_refit = 0

        self._committee = [
            LogisticRegression(max_iter=1000, solver="lbfgs", warm_start=True),
            DecisionTreeClassifier(max_depth=6, random_state=42),
            GaussianNB(),
        ]

        self._build_feature_map()

    # Initialisation 

    def _build_feature_map(self):
        df = pd.read_csv(self.action_space_file, delimiter="|", index_col=False)
        df["categories"] = df["categories"].apply(literal_eval)

        for _, row in df.iterrows():
            domain = row["domain"]
            cats   = row["categories"] if isinstance(row["categories"], list) else []
            entity = row["entity"] if pd.notna(row.get("entity")) else ""
            rank   = int(row["rank"])
            self._feature_map[domain] = _build_feature_vector(cats, entity, rank)

    def reset(self):
        super().reset()
        self._labels.clear()
        self._committee_fitted = False
        self._probes_since_refit = 0
        self._committee = [
            LogisticRegression(max_iter=1000, solver="lbfgs", warm_start=True),
            DecisionTreeClassifier(max_depth=6, random_state=42),
            GaussianNB(),
        ]

    #  Committee training 

    def _fit_committee(self):
        self._probes_since_refit += 1
        if self._probes_since_refit < REFIT_EVERY and self._committee_fitted:
            return

        if len(self._labels) < MIN_SAMPLES_TO_FIT:
            return

        domains = [d for d in self._labels if d in self._feature_map]
        if not domains:
            return

        X = np.array([self._feature_map[d] for d in domains])
        y = np.array([self._labels[d]       for d in domains])

        if len(np.unique(y)) < 2:
            return

        for clf in self._committee:
            clf.fit(X, y)

        self._committee_fitted = True
        self._probes_since_refit = 0

    #  Selection 

    def _select_domain(self) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
        candidates      = []
        candidate_names = []

        for node, n_data in self.action_space.gen_active_target_nodes_and_data():
            candidates.append(node)
            candidate_names.append(n_data[action_space_module.NAME])

        if not candidates:
            return None, None

        if not self._committee_fitted:
            idx = random.randint(0, len(candidates) - 1)
            return candidates[idx], candidate_names[idx]

        feat_indices = [i for i, n in enumerate(candidate_names) if n in self._feature_map]

        if not feat_indices:
            idx = random.randint(0, len(candidates) - 1)
            return candidates[idx], candidate_names[idx]

        X_cands = np.array([self._feature_map[candidate_names[i]] for i in feat_indices])

        # collect each committee member's block probability for all candidates
        probs_matrix = np.array([
            clf.predict_proba(X_cands)[:, 1] for clf in self._committee
        ])  # shape: (n_committee, n_candidates)

        entropy      = _vote_entropy(probs_matrix)
        best_local   = int(np.argmax(entropy))
        best_global  = feat_indices[best_local]

        return candidates[best_global], candidate_names[best_global]

    #  Model interface 

    def choose_arm(self) -> typing.List[str]:
        raise NotImplementedError("QueryByCommittee selects targets globally via step()")

    def observe(self, selected_arm: str, measurement_result: float) -> float:
        return measurement_result

    def step(self) -> dict:
        selected_target, selected_target_name = self._select_domain()

        if selected_target is None:
            return {
                "action": "none", "target": "none",
                "reward": 0.0, "q_value": 0.0,
                "is_blocked": 0, "is_optimal": 0,
                "coverage": self.get_blocklist_coverage(),
            }

        measurement_result, is_blocked = self.take_measurement(selected_target_name)

        if is_blocked:
            self.update_blocklist_target_found(selected_target_name)

        self._labels[selected_target_name] = 1 if is_blocked else 0
        self._fit_committee()
        self.disable_target(selected_target)

        return {
            "action":     "query_by_committee",
            "target":     selected_target_name,
            "reward":     round(measurement_result, 2),
            "q_value":    round(float(is_blocked), 2),
            "is_blocked": 1 if is_blocked else 0,
            "is_optimal": 0,
            "coverage":   self.get_blocklist_coverage(),
        }


class QbCParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()

    def set_params(self, args):
        super().set_params(args)
        self.params["action_value_file"] = None


if __name__ == "__main__":
    parser = QbCParserOptions()
    params = parser.parse()
    run_multiprocessing(QueryByCommittee, params)
