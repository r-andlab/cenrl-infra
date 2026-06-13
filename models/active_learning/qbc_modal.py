"""
Query by Committee using the modAL open-source active learning library.

modAL (https://github.com/modAL-python/modAL) 

  - Logistic Regression  (linear decision boundary)
  - Decision Tree        (non-linear, axis-aligned splits)
  - Naive Bayes          (feature-independent probabilistic model)

Reference:
  Danka, T. & Horvath, P. (2018). modAL: A Modular Active Learning Framework
  for Python. ICLR Workshop. https://github.com/modAL-python/modAL

Usage:
    python3 models/active_learning/qbc_modal.py \\
        -m 1000 -E 3 \\
        -o outputs/qbc_modal \\
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
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import consensus_entropy_sampling

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

MIN_SAMPLES_TO_FIT = 20


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


def _make_committee() -> Committee:
    learners = [
        ActiveLearner(estimator=LogisticRegression(max_iter=1000, solver="lbfgs")),
        ActiveLearner(estimator=DecisionTreeClassifier(max_depth=6, random_state=42)),
        ActiveLearner(estimator=GaussianNB()),
    ]
    return Committee(learner_list=learners, query_strategy=consensus_entropy_sampling)


class QbCModAL(Model):
    """
    QbC using modAL's Committee with vote_entropy_sampling.
    committee taught incrementally after each probe via committee.teach().
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

        self._feature_map: typing.Dict[str, np.ndarray] = {}
        self._labels: typing.Dict[str, int] = {}
        self._committee: typing.Optional[Committee] = None
        self._committee_fitted = False

        self._build_feature_map()
    
    #  Initialisation 
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
        self._committee = None
        self._committee_fitted = False

    #  Committee training 

    def _update_committee(self, domain: str, label: int):
        """
        Incrementally teach the committee one new labeled example.
        first call with enough diversity, initialises the committee via fit.
        Next calls use committee.teach() for incremental updates.
        """
        if domain not in self._feature_map:
            return

        if len(self._labels) < MIN_SAMPLES_TO_FIT:
            return

        # need at least one positive and one negative
        if len(set(self._labels.values())) < 2:
            return

        x = self._feature_map[domain].reshape(1, -1)
        y = np.array([label])

        if not self._committee_fitted:
            # Bootstrap: create committee with training data passed to each learner
            domains = [d for d in self._labels if d in self._feature_map]
            X = np.array([self._feature_map[d] for d in domains])
            Y = np.array([self._labels[d]       for d in domains])
            self._committee = Committee(
                learner_list=[
                    ActiveLearner(estimator=LogisticRegression(max_iter=1000, solver="lbfgs"),
                                  X_training=X, y_training=Y),
                    ActiveLearner(estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
                                  X_training=X, y_training=Y),
                    ActiveLearner(estimator=GaussianNB(),
                                  X_training=X, y_training=Y),
                ],
                query_strategy=consensus_entropy_sampling
            )
            self._committee_fitted = True
        else:
            # Incremental update with the single new example
            self._committee.teach(x, y)

    # Selection 

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

        # Compute vote entropy via predict_proba (matches plain QueryByCommittee).
        # Using committee.query() with vote_entropy_sampling uses hard predict() votes, which collapse to zero entropy when the committee unanimously predicts "not blocked" 
        probs_matrix = np.array([
            learner.predict_proba(X_cands)[:, 1]
            for learner in self._committee.learner_list
        ])
        p = np.clip(probs_matrix.T, 1e-9, 1 - 1e-9)
        entropies = (-(p * np.log(p) + (1 - p) * np.log(1 - p))).mean(axis=1)
        best_local  = int(np.argmax(entropies))
        best_global = feat_indices[best_local]

        return candidates[best_global], candidate_names[best_global]

    #  Model interface 

    def choose_arm(self) -> typing.List[str]:
        raise NotImplementedError("QbCModAL selects targets globally via step()")

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
        self._update_committee(selected_target_name, 1 if is_blocked else 0)
        self.disable_target(selected_target)

        return {
            "action":     "qbc_modal",
            "target":     selected_target_name,
            "reward":     round(measurement_result, 2),
            "q_value":    round(float(is_blocked), 2),
            "is_blocked": 1 if is_blocked else 0,
            "is_optimal": 0,
            "coverage":   self.get_blocklist_coverage(),
        }


class QbCModALParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()

    def set_params(self, args):
        super().set_params(args)
        self.params["action_value_file"] = None


if __name__ == "__main__":
    parser = QbCModALParserOptions()
    params = parser.parse()
    run_multiprocessing(QbCModAL, params)
