"""
LinUCB with a single shared (global) A matrix and b vector across all arms.

In the standard (disjoint) LinUCB each arm maintains independent A_a and b_a,
so learning from one arm never updates another. Here, every probe updates one
global A and b regardless of which arm was chosen:

    A_global  +=  x_a x_a^T        (updated after every probe)
    b_global  +=  r_t * x_a

    θ_global   =  A_global⁻¹ b_global

    score(a)   =  θ_global · x_a  +  α * sqrt(x_a · A_global⁻¹ · x_a)

This means probing "Pornography" and seeing reward=1 immediately raises the
predicted score for every arm that shares features with it (e.g. Adult Themes,
Dating & Relationships) — true cross-arm knowledge transfer.
"""

import typing
from random import randrange

import numpy as np

import models.base.action_space as action_space_module
from models.base.action_space import (
    NODE_TYPE_KEY, POSSIBLE_TARGET_FEATURES,
    ACTION_ATTEMPTS, Q_VALUE, IS_TARGET_NODE
)
from models.base.model import Model, ParserOptions, run_multiprocessing
from models.contextual_bandit.linucb import (
    LinUCBActionSpace, FEATURE_DIM, _static_features
)

np.seterr(divide="ignore", invalid="ignore")


class LinUCBShared(Model):
    """
    LinUCB with a single global A/b shared across all arms.
    Arm selection and hierarchy traversal are identical to LinUCB;
    only the update rule changes.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.alpha = params["alpha"]
        self.initial_value_estimate = params["initial_value_estimate"]

        # One global matrix and vector — shared across ALL arms
        self._A_global = np.eye(FEATURE_DIM)
        self._b_global = np.zeros(FEATURE_DIM)

        self.exploration_epoch_num = self.current_epoch_num

    def reset(self):
        super().reset()
        self._A_global = np.eye(FEATURE_DIM)
        self._b_global = np.zeros(FEATURE_DIM)
        self.exploration_epoch_num = self.current_epoch_num

    # ── Feature construction (identical to LinUCB) ────────────────────────────

    def _get_feature_vector(self, node_key: str) -> np.ndarray:
        n_data   = self.action_space.get(node_key)
        node_name = n_data[action_space_module.NAME]

        static = _static_features(node_name)

        total    = n_data.get("total_domains", 1)
        log_size = np.log(total + 1) / np.log(10_001)

        graph  = self.action_space.get_graph()
        active = sum(
            1 for c in graph.successors(node_key)
            if graph.nodes[c].get(IS_TARGET_NODE, False)
               and not graph.nodes[c].get(action_space_module.SLEEPING, False)
        )
        fraction_remaining = active / total

        return np.concatenate([static, [log_size, fraction_remaining]])

    # ── Scoring using global weights ──────────────────────────────────────────

    def _score(self, node_key: str) -> float:
        x          = self._get_feature_vector(node_key)
        theta      = np.linalg.solve(self._A_global, self._b_global)
        predicted  = float(theta @ x)
        A_inv_x    = np.linalg.solve(self._A_global, x)
        uncertainty = float(np.sqrt(x @ A_inv_x))
        return predicted + self.alpha * uncertainty

    # ── Arm selection (same traversal logic as LinUCB) ────────────────────────

    def choose_arm(self) -> typing.List[str]:
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
                scores.append(self._score(succ))

            if reached_target_nodes:
                break

            if immediate_children:
                scores_arr = np.array(scores)
                top_indices = np.argwhere(scores_arr == scores_arr.max()).flatten()
                selected_arm = immediate_children[top_indices[randrange(len(top_indices))]]
                selected_arms_history.append(selected_arm)

            source = selected_arm

        for a in selected_arms_history[:-1]:
            self.action_space.get(a)[ACTION_ATTEMPTS] += 1

        self.last_selected_arm_index = selected_arms_history[-1]
        return selected_arms_history

    # ── Global weight update ──────────────────────────────────────────────────

    def observe(self, selected_arm: str, measurement_result: float) -> float:
        x = self._get_feature_vector(selected_arm)

        # Update the single global model with this probe
        self._A_global += np.outer(x, x)
        self._b_global += measurement_result * x

        # Derive predicted reward from updated global weights
        theta     = np.linalg.solve(self._A_global, self._b_global)
        predicted = float(np.clip(theta @ x, 0.0, 1.0))

        n_data = self.action_space.get(selected_arm)
        n_data[ACTION_ATTEMPTS] += 1
        n_data[Q_VALUE] = round(predicted, 2)

        return n_data[Q_VALUE]

    def step(self) -> dict:
        self.exploration_epoch_num += 1
        return super().step()


class LinUCBSharedParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()

    def set_params(self, args):
        super().set_params(args)
        self.params["alpha"] = 0.5
        self.params["initial_value_estimate"] = 0.0
        self.params["action_value_file"] = None


if __name__ == "__main__":
    parser = LinUCBSharedParserOptions()
    params = parser.parse()
    run_multiprocessing(LinUCBShared, params)
