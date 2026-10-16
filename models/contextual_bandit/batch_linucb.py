"""
batch_linucb.py

BatchLinUCB / BatchClusterLinUCB -- the batching/async layer BatchUCB
(Infrastructure/models/BatchUCB.py) adds on top of UCBNaive, retargeted to
drive a LinUCB-family model instead.

Why this exists
----------------
node.py's live orchestration flow never calls choose_arm()/observe()
directly -- it calls queue_measurement(), register_batch(), can_step(),
absorb_measurement(), flush_held_batches(). Those five methods are ALL
BatchUCB-specific; UCBNaive alone doesn't have them, and neither does
LinUCB/ClusterLinUCB (both only expose the simple synchronous
choose_arm()/observe()/step()/reset() interface built for the standalone
offline simulation loop in run_multiprocessing, driven by scripts like
scripts/eval_full/run_ucb_linucb_rerun.py).

Everything BatchUCB's batch layer calls OTHER than choose_arm()/observe()
-- is_optimal_action, propagate_rewards, disable_target,
update_optimal_value, can_step -- lives on the shared Model base class both
UCBNaive and LinUCB inherit from (models/base/model.py). So this batch
layer needed zero logic changes to work against LinUCB's identically-shaped
choose_arm()/observe(): LinUCB.choose_arm() is explicitly documented
"identical to UCBNaive.choose_arm()", and observe(selected_arm: str,
measurement_result: float) -> float matches signature-for-signature.

Built as a mixin (BatchLinUCBMixin) rather than editing BatchUCB.py
directly, so:
  1. BatchUCB.py -- the live, validated pipeline every run this session has
     used -- is never at risk of being left in a half-edited state.
  2. The mixin combines with EITHER LinUCB or ClusterLinUCB via multiple
     inheritance, so both models get this layer from ONE implementation.
     ClusterLinUCB only overrides internal scoring/update (_linucb_score,
     observe's arithmetic) -- it doesn't touch choose_arm()'s outer
     traversal structure at all, so no separate mixin is needed for it.

This is a near-verbatim adaptation of BatchUCB.py's own logic -- same
target-selection strategies (TOP_K_FROM_ARM / UNIFORM_SPREAD /
WEIGHTED_SPREAD), same batch-size and reward-propagation options
(CONSTANT_VAL / VARY_ON_SUCCESS, ON_RECEIPT / IN_ORDER) -- kept identical
on purpose so a run using this is comparable to a BatchUCB run under the
same settings, differing only in the underlying arm-scoring algorithm.

Also includes the same no-op ground-truth-file overrides BatchUCB.py has
(parse_block_list / init_blockers / etc.) -- LinUCB/ClusterLinUCB inherit
Model's versions unmodified, which require a real blocklist CSV to
construct at all; live orchestrator-driven runs compute reward externally
from real Hyperquack measurements and never touch this machinery, exactly
like BatchUCB today.
"""

import logging
from typing import Dict, List

import models.base.action_space as action_space_module
from Infrastructure.utils.structures import (
    BatchSizeMethod,
    BatchSelectionMethod,
    PropagationMethod,
    Pending,
)
from common.utils import LOG_FILE_DELIMITER
from models.contextual_bandit.linucb import LinUCB
from models.contextual_bandit.cluster_linucb import ClusterLinUCB

logger = logging.getLogger(__name__)


class BatchLinUCBMixin:
    """Batch/async layer -- see module docstring. Mix in BEFORE a LinUCB-family
    class, e.g. `class BatchLinUCB(BatchLinUCBMixin, LinUCB)`, so this
    class's queue_measurement/absorb_measurement/etc. take priority in the
    MRO, while choose_arm()/observe()/reset()/step() fall through to the
    LinUCB-family class unchanged."""

    def __init__(
        self,
        params,
        country_name,
        target_selection: BatchSelectionMethod = BatchSelectionMethod.TOP_K_FROM_ARM,
        batch_size_method: BatchSizeMethod = BatchSizeMethod.CONSTANT_VAL,
        qval_propagation_method: PropagationMethod = PropagationMethod.ON_RECEIPT,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._selected_targets: Dict[str, Pending] = {}
        self.selection_method = target_selection
        self.size_method = batch_size_method
        self.prop_method = qval_propagation_method
        self.country_name = country_name

        # Same IN_ORDER hold-buffer bookkeeping as BatchUCB.py -- see that
        # file's __init__ comment for the full invariant description.
        self._held_batches: Dict[int, list] = {}
        self._batch_target_set: Dict[int, set] = {}
        self._batch_decremented: Dict[int, int] = {}
        self._target_to_batch: Dict[str, int] = {}

    def register_batch(self, batch_id: int, target_names: List[str]) -> None:
        if self.prop_method is not PropagationMethod.IN_ORDER:
            return
        if not target_names:
            return
        self._held_batches[batch_id] = []
        self._batch_target_set[batch_id] = set(target_names)
        self._batch_decremented[batch_id] = 0
        for t_name in target_names:
            self._target_to_batch[t_name] = batch_id

    def choose_targets(
        self, selected_arm_key: str, selected_arm_name: str, selection_size: int = 10
    ) -> List[str]:
        if self.selection_method is BatchSelectionMethod.TOP_K_FROM_ARM:
            chosen_targets = self.action_space.sample_successors(
                selected_arm_key,
                n_samples=selection_size,
                use_rank_weights=self.sample_by_target_rank,
            )
            return chosen_targets[0:selection_size]
        elif self.selection_method in (
            BatchSelectionMethod.UNIFORM_SPREAD,
            BatchSelectionMethod.WEIGHTED_SPREAD,
        ):
            raise NotImplementedError(
                "SPREAD callers must use queue_measurement; choose_targets is per-arm only"
            )
        return []

    def _eligible_leaf_arms(self) -> List[tuple]:
        seen_arms: Dict[str, float] = {}
        graph = self.action_space.get_graph()
        for leaf_node, _n_data in self.action_space.gen_active_target_nodes_and_data():
            for parent in graph.predecessors(leaf_node):
                if parent in seen_arms:
                    continue
                arm_data = self.action_space.get(parent)
                seen_arms[parent] = arm_data[action_space_module.Q_VALUE]
        return list(seen_arms.items())

    def _compute_arm_allocations(
        self, arms: List[tuple], k: int, method: BatchSelectionMethod
    ) -> Dict[str, int]:
        import numpy as np

        if not arms or k <= 0:
            return {}

        n_arms = len(arms)
        arm_keys = [a for a, _q in arms]
        q_values = np.array([q for _a, q in arms], dtype=float)

        if method is BatchSelectionMethod.UNIFORM_SPREAD:
            base_share = k // n_arms
            leftover = k % n_arms
            allocations = [base_share] * n_arms
            order = sorted(range(n_arms), key=lambda i: -q_values[i])
            for j in range(leftover):
                allocations[order[j]] += 1
        elif method is BatchSelectionMethod.WEIGHTED_SPREAD:
            temperature = 1.0
            shifted = (q_values - q_values.max()) / temperature
            exps = np.exp(shifted)
            weights = exps / exps.sum()
            float_allocs = weights * k
            floor_allocs = np.floor(float_allocs).astype(int)
            leftover = k - int(floor_allocs.sum())
            allocations = floor_allocs.tolist()
            if leftover > 0:
                order = sorted(range(n_arms), key=lambda i: -weights[i])
                for j in range(leftover):
                    allocations[order[j]] += 1
        else:
            return self._compute_arm_allocations(arms, k, BatchSelectionMethod.UNIFORM_SPREAD)

        return {arm_keys[i]: int(allocations[i]) for i in range(n_arms) if allocations[i] > 0}

    def _arm_seq_for_leaf(self, leaf_arm_key: str) -> List[str]:
        graph = self.action_space.get_graph()
        root = self.action_space.get_root()
        seq: List[str] = []
        node = leaf_arm_key
        while node is not None and node != root:
            seq.append(node)
            parents = list(graph.predecessors(node))
            assert len(parents) <= 1, (
                f"action-space DAG has multi-parent node: {node} "
                f"(parents={parents}); single-parent-walk invariant violated"
            )
            if not parents:
                break
            node = parents[0]
        seq.reverse()
        return seq

    def _queue_measurement_spread(self, batch_size: int) -> List[str]:
        eligible = self._eligible_leaf_arms()
        if not eligible:
            return []

        allocations = self._compute_arm_allocations(eligible, batch_size, self.selection_method)
        if not allocations:
            return []

        new_targets: Dict[str, Pending] = {}
        for arm_key, allocation in allocations.items():
            sampled_node_ids = self.action_space.sample_successors(
                arm_key,
                n_samples=allocation,
                use_rank_weights=self.sample_by_target_rank,
            )
            if not sampled_node_ids:
                continue
            arm_seq = self._arm_seq_for_leaf(arm_key)
            for node_id in sampled_node_ids[:allocation]:
                t_name = self.action_space.get(node_id)[action_space_module.NAME]
                if t_name in self._selected_targets or t_name in new_targets:
                    continue
                new_targets[t_name] = Pending(
                    node_id=node_id,
                    arm_key=arm_key,
                    arm_seq=arm_seq,
                )

        self._selected_targets.update(new_targets)

        if self.verbose and self.logfile:
            arms_used = sorted(allocations.keys())
            self.logfile.write(
                f"SPREAD arms={len(arms_used)} alloc={allocations} k={batch_size}"
                + LOG_FILE_DELIMITER
                + f"Selection Size {len(new_targets)}"
                + LOG_FILE_DELIMITER
            )

        return list(new_targets.keys())

    def queue_measurement(self, batch_size) -> List[str]:
        if self.selection_method is BatchSelectionMethod.TOP_K_FROM_ARM:
            arm_seq = self.choose_arm()
            arm_key = arm_seq[-1]
            arm_name = self.action_space.get(arm_key)[action_space_module.NAME]

            if self.verbose and self.logfile:
                self.logfile.write(str(arm_name) + LOG_FILE_DELIMITER)

            node_ids = self.choose_targets(arm_key, arm_name, batch_size)

            new_targets: Dict[str, Pending] = {}
            for node_id in node_ids:
                t_name = self.action_space.get(node_id)[action_space_module.NAME]
                if t_name in self._selected_targets or t_name in new_targets:
                    continue

                new_targets[t_name] = Pending(
                    node_id=node_id,
                    arm_key=arm_key,
                    arm_seq=arm_seq,
                )

            self._selected_targets.update(new_targets)

            if self.verbose and self.logfile:
                self.logfile.write(f"Selection Size {batch_size}" + LOG_FILE_DELIMITER)

            return list(new_targets.keys())

        return self._queue_measurement_spread(batch_size)

    def absorb_measurement(self, result: Dict[str, str]):
        t_name = result["target"]
        if t_name not in self._selected_targets.keys():
            raise KeyError(f"{self.country_name}: Received results from unknown target {t_name}")

        pending: Pending = self._selected_targets[t_name]

        is_blocked = result["blocked"]
        # Ex 6: reward_mode picks what number counts as the reward. is_blocked
        # (and everything derived from it below -- the CSV is_blocked column,
        # update_blocklist_target_found) stays the boolean majority verdict
        # regardless of reward_mode; only the reward itself changes. Mirrors
        # BatchUCB.absorb_measurement's identical branch (Infrastructure/
        # models/BatchUCB.py) -- this mixin had no reward_mode handling at
        # all before, so every LinUCB-family run silently got binary reward
        # even when --reward-mode percentage was passed.
        reward_mode = getattr(self, "reward_mode", "majority")
        if reward_mode == "percentage":
            reward = float(result.get("vote_fraction", 1.0 if is_blocked else 0.0))
        else:
            reward = 1.0 if is_blocked else 0.0
        pending.reward = reward
        pending.blocked = is_blocked

        if is_blocked:
            self.update_blocklist_target_found(t_name)

        if self.verbose and self.logfile:
            self.logfile.write(
                t_name + LOG_FILE_DELIMITER + str(int(reward)) + "\n"
            )

        arm_seq = pending.arm_seq
        arm_key = pending.arm_key

        is_optimal = self.is_optimal_action(arm_seq)
        observed_value = self.observe(arm_key, reward)

        assert observed_value is not None, "Missing observed value, expecting a float"
        pending.observed_value = observed_value

        self.update_optimal_value()

        for n in arm_seq + [pending.node_id]:
            self.action_space.get(n)[action_space_module.EXPLORED] = True

        return_value = {
            "action": arm_key,
            "targets": t_name,
            "rewards": round(pending.reward, 2),
            "q_value": round(pending.observed_value, 2),
            "is_blocked": 1 if is_blocked else 0,
            "is_optimal": 1 if is_optimal else 0,
            "coverage": self.get_blocklist_coverage(),
        }

        if self.prop_method is PropagationMethod.ON_RECEIPT:
            self.propagate_rewards(arm_key)
            self.disable_target(pending.node_id)
            del self._selected_targets[t_name]
        elif self.prop_method is PropagationMethod.IN_ORDER:
            batch_id = self._target_to_batch.get(t_name)
            if batch_id is None:
                logger.warning(
                    "%s: IN_ORDER absorb for unregistered target %s -- falling back to ON_RECEIPT",
                    self.country_name, t_name,
                )
                self.propagate_rewards(arm_key)
                self.disable_target(pending.node_id)
                del self._selected_targets[t_name]
            else:
                self._held_batches[batch_id].append((t_name, pending, return_value))
                self._release_complete_batches()
        else:
            logger.error(
                "%s: unknown prop_method %s -- falling back to ON_RECEIPT",
                self.country_name, self.prop_method,
            )
            self.propagate_rewards(arm_key)
            self.disable_target(pending.node_id)
            del self._selected_targets[t_name]

        return return_value

    def _release_complete_batches(self) -> None:
        if not self._held_batches:
            return
        for batch_id in sorted(self._held_batches.keys()):
            target_set = self._batch_target_set.get(batch_id)
            if target_set is None:
                break
            absorbed_count = len(self._held_batches[batch_id])
            decremented_count = self._batch_decremented.get(batch_id, 0)
            if absorbed_count + decremented_count < len(target_set):
                break
            held = self._held_batches.pop(batch_id)
            self._batch_target_set.pop(batch_id, None)
            self._batch_decremented.pop(batch_id, None)
            for t_name, pending, _return_value in held:
                arm_key = pending.arm_key
                self.propagate_rewards(arm_key)
                self.disable_target(pending.node_id)
                self._selected_targets.pop(t_name, None)
                self._target_to_batch.pop(t_name, None)

    def flush_held_batches(self) -> int:
        if not self._held_batches:
            return 0
        released = 0
        for batch_id in sorted(self._held_batches.keys()):
            held = self._held_batches[batch_id]
            for t_name, pending, _return_value in held:
                try:
                    self.propagate_rewards(pending.arm_key)
                    self.disable_target(pending.node_id)
                except Exception as exc:
                    logger.error(
                        "%s: flush_held_batches: propagate failed for target %s: %s",
                        self.country_name, t_name, exc,
                    )
                self._selected_targets.pop(t_name, None)
                self._target_to_batch.pop(t_name, None)
                released += 1
        self._held_batches.clear()
        self._batch_target_set.clear()
        self._batch_decremented.clear()
        return released

    # ── Ground-truth no-ops (same reasoning as BatchUCB.py) ────────────────
    def parse_block_list(self):
        pass

    def get_blocklist_coverage(self) -> float:
        return 0

    def set_blocklist_unique_counts_based_on_action_space(self):
        pass

    def init_blockers(self):
        pass

    def update_blocklist_target_found(self, target_found: str):
        pass


class BatchLinUCB(BatchLinUCBMixin, LinUCB):
    """Plain disjoint LinUCB, live-orchestrator-compatible."""
    pass


class BatchClusterLinUCB(BatchLinUCBMixin, ClusterLinUCB):
    """ClusterLinUCB (semantic cluster oracle on top of LinUCB), live-orchestrator-compatible."""
    pass
