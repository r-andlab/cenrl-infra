from models.ucb.ucb_naive import UCBNaive
import models.base.action_space as action_space_module
from typing import List, Dict
import numpy as np
from Infrastructure.utils.structures import (
    BatchSizeMethod,
    BatchSelectionMethod,
    PropagationMethod,
    Pending
)
from common.utils import (
    LOG_FILE_DELIMITER,
    NO_DATE_BLOCKLIST,
    TARGET_FEATURE__DOMAIN,
    UNKNOWN_EMPTY,
    TARGET_FEATURE__SERVICE_IP,
    SERVER_FEATURE_ORDER,
)


class BatchUCB(UCBNaive):
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

    def choose_targets(
        self, selected_arm_key: str, selected_arm_name: str, selection_size: int = 10
    ) -> List[str]:
        """Select target node_ids for a given arm.

        TOP_K_FROM_ARM (default): sample K via sample_successors from `selected_arm_key`.

        UNIFORM_SPREAD / WEIGHTED_SPREAD: NOT supported via this per-arm API.
        Per D-14, SPREAD modes allocate K across MULTIPLE leaf arms — they
        cannot produce a correct per-target `arm_seq` from a single
        `selected_arm_key`. The single legitimate SPREAD entry path is
        `queue_measurement` -> `_queue_measurement_spread`, which builds
        Pending objects with the correct per-leaf arm_seq via the
        predecessors walk.

        W4 fix: previously this path silently delegated to a fallback that
        produced incorrect Pendings (no arm_seq). That fallback is removed;
        misuse raises loudly.
        """
        if self.selection_method is BatchSelectionMethod.TOP_K_FROM_ARM:
            """Method: Choose top-k from chosen arm"""
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
                "SPREAD callers must use queue_measurement; choose_targets is "
                "per-arm only — see D-14"
            )
        return []

    def _eligible_leaf_arms(self) -> List[tuple]:
        """D-10: enumerate eligible leaf arms = arms with >=1 non-SLEEPING target successor.

        Returns: list of (arm_key, q_value) pairs. Insertion order = NetworkX
        descendant order from root, which gives a stable tie-break for
        leftover allocation (D-11/D-12).

        A "leaf arm" is the lowest-level non-target node — its direct successors
        are IS_TARGET_NODE leaves. We enumerate the active leaf-target population
        (gen_active_target_nodes_and_data, action_space.py:344) and lift each one
        to its parent arm via NetworkX predecessors. This guarantees we only see
        arms with >=1 awake target.
        """
        seen_arms: Dict[str, float] = {}
        graph = self.action_space.get_graph()
        for leaf_node, _n_data in self.action_space.gen_active_target_nodes_and_data():
            # Each active leaf has exactly one parent in this DAG (per action_space construction).
            for parent in graph.predecessors(leaf_node):
                if parent in seen_arms:
                    continue
                arm_data = self.action_space.get(parent)
                seen_arms[parent] = arm_data[action_space_module.Q_VALUE]
        # Stable ordering = arm insertion order (dict preserves insertion since 3.7).
        return list(seen_arms.items())

    def _compute_arm_allocations(
        self, arms: List[tuple], k: int, method: BatchSelectionMethod
    ) -> Dict[str, int]:
        """D-11/D-12: Hare/Hamilton apportionment of K targets across `arms`.

        UNIFORM (D-11): equal floor share + leftover to highest-Q arms.
        WEIGHTED (D-12): softmax(Q/T) shares + leftover to highest-weight arms.

        Both share the same leftover-distribution rule: sort by descending
        weight (UNIFORM uses Q-value; WEIGHTED uses softmax weight); leftover
        targets go to the top of that ordering, with insertion order as the
        secondary key (stable Python sort gives this for free).

        Zero-allocation arms (D-13): not included in the returned dict at all
        (callers must skip arms not in the dict). No minimum-1-per-arm
        guarantee — coverage is the bandit's outer-loop responsibility.

        Returns: {arm_key: allocation_count} for arms with >=1 allocation.
        """
        if not arms or k <= 0:
            return {}

        n_arms = len(arms)
        arm_keys = [a for a, _q in arms]
        q_values = np.array([q for _a, q in arms], dtype=float)

        if method is BatchSelectionMethod.UNIFORM_SPREAD:
            # D-11: equal floor share, leftover to highest-Q arms.
            base_share = k // n_arms
            leftover = k % n_arms
            allocations = [base_share] * n_arms
            # Sort indices by descending Q (stable, so insertion order breaks ties).
            order = sorted(range(n_arms), key=lambda i: -q_values[i])
            for j in range(leftover):
                allocations[order[j]] += 1
        elif method is BatchSelectionMethod.WEIGHTED_SPREAD:
            # D-12: softmax(Q/T), default temperature = 1.0 (hard-coded constant per CONTEXT.md).
            # Pitfall 8 mitigation: log-sum-exp shift via subtracting Q.max() before exp,
            # so the largest exp argument is 0.0 (no overflow even at extreme Q values).
            temperature = 1.0
            shifted = (q_values - q_values.max()) / temperature
            exps = np.exp(shifted)
            weights = exps / exps.sum()  # stable softmax in [0,1]
            # Floor allocation per arm + leftover to highest-weight arms.
            float_allocs = weights * k
            floor_allocs = np.floor(float_allocs).astype(int)
            leftover = k - int(floor_allocs.sum())
            allocations = floor_allocs.tolist()
            if leftover > 0:
                # Distribute leftover to highest-weight arms (insertion-order tie-break via stable sort).
                order = sorted(range(n_arms), key=lambda i: -weights[i])
                for j in range(leftover):
                    allocations[order[j]] += 1
        else:
            # Defensive: should be unreachable because the caller already
            # branched. Fall back to UNIFORM rather than raise.
            return self._compute_arm_allocations(arms, k, BatchSelectionMethod.UNIFORM_SPREAD)

        # D-13: skip zero-allocation arms by omitting them from the dict.
        return {arm_keys[i]: int(allocations[i]) for i in range(n_arms) if allocations[i] > 0}

    def _arm_seq_for_leaf(self, leaf_arm_key: str) -> List[str]:
        """D-14: rebuild arm_seq for a chosen leaf arm by walking parents up to root.

        Mirrors the shape that `choose_arm()` returns for non-SPREAD modes:
        a list of arm keys from root-child down to the leaf arm (root EXCLUDED,
        leaf included). The downstream `is_optimal_action`/`propagate_rewards`
        consumers iterate this list and skip the root themselves
        (model.py:168 root-skip in propagate_rewards), so excluding root here
        matches the existing convention.

        Implementation: NetworkX predecessors walk (model.py:181 idiom);
        each non-target node has exactly one parent in this DAG so each step
        is unambiguous.

        W3 fix: assert single-parent invariant (D-14: action-space DAG is a tree
        by construction). A future graph topology change that introduced
        multi-parent nodes would silently corrupt arm_seq via the bare
        `parents[0]` selection — propagating rewards to the wrong arm. The
        assertion converts that silent corruption into a loud failure at the
        point of the bad assumption.
        """
        graph = self.action_space.get_graph()
        root = self.action_space.get_root()
        seq: List[str] = []
        node = leaf_arm_key
        # Walk upward; stop when we reach root (do not include root itself).
        while node is not None and node != root:
            seq.append(node)
            parents = list(graph.predecessors(node))
            # W3: D-14 single-parent-walk assumption made explicit.
            assert len(parents) <= 1, (
                f"action-space DAG has multi-parent node: {node} "
                f"(parents={parents}); D-14 single-parent-walk invariant violated"
            )
            if not parents:
                break  # disconnected (shouldn't happen for an eligible arm)
            node = parents[0]  # single-parent DAG by construction
        # Reverse so order is root-child -> leaf, matching choose_arm() shape.
        seq.reverse()
        return seq

    def _queue_measurement_spread(self, batch_size: int) -> List[str]:
        """D-14/D-15/D-16: SPREAD entry path. Per D-14 the per-arm chooser is
        bypassed entirely; targets are allocated directly across eligible leaf arms.

        For each eligible leaf arm:
          1. Read Q-value via _eligible_leaf_arms (cheap; one pass).
          2. Allocate K via Hare/Hamilton (UNIFORM equal share or WEIGHTED softmax).
          3. Sample `allocation` targets via sample_successors with rank-weighting
             matching self.sample_by_target_rank (D-15: reuse existing call).
          4. Build a Pending per target with arm_key = leaf arm and arm_seq =
             _arm_seq_for_leaf walk (D-14).
          5. Apply the existing dedup against self._selected_targets (preserves
             SPREAD's per-batch-window safety).

        Short-batch (D-16): if the sum of per-arm samples is M < K, return
        what was found. The orchestrator's in_flight accounting and the
        iteration-quota guard already handle short batches.
        """
        eligible = self._eligible_leaf_arms()
        if not eligible:
            return []

        allocations = self._compute_arm_allocations(eligible, batch_size, self.selection_method)
        if not allocations:
            return []

        new_targets: Dict[str, Pending] = {}
        for arm_key, allocation in allocations.items():
            # D-15: reuse existing sample_successors with the per-arm allocation.
            sampled_node_ids = self.action_space.sample_successors(
                arm_key,
                n_samples=allocation,
                use_rank_weights=self.sample_by_target_rank,
            )
            if not sampled_node_ids:
                continue
            # D-14: per-target arm_seq from leaf arm upward (one walk per arm; reused for all this arm's targets).
            arm_seq = self._arm_seq_for_leaf(arm_key)
            for node_id in sampled_node_ids[:allocation]:
                t_name = self.action_space.get(node_id)[action_space_module.NAME]
                # Per-batch-window dedup (matches BatchUCB.py:73 idiom).
                if t_name in self._selected_targets or t_name in new_targets:
                    continue
                new_targets[t_name] = Pending(
                    node_id=node_id,
                    arm_key=arm_key,
                    arm_seq=arm_seq,
                )

        self._selected_targets.update(new_targets)

        if self.verbose and self.logfile:
            # SPREAD per-batch summary (Claude's discretion in CONTEXT.md):
            # write "SPREAD" tag plus arm count and total selection size.
            arms_used = sorted(allocations.keys())
            self.logfile.write(
                f"SPREAD arms={len(arms_used)} alloc={allocations} k={batch_size}"
                + LOG_FILE_DELIMITER
                + f"Selection Size {len(new_targets)}"
                + LOG_FILE_DELIMITER
            )

        # D-16: return what was found, even if M < batch_size.
        return list(new_targets.keys())

    def queue_measurement(self, batch_size) -> list[str]:
        """
        Selects targets according to self.selection_method.

        TOP_K_FROM_ARM (default): pick one arm via choose_arm(), sample K targets from it.
        UNIFORM_SPREAD / WEIGHTED_SPREAD (D-14): bypass choose_arm(); allocate K
        across all eligible leaf arms via _compute_arm_allocations and sample
        per-arm via sample_successors.
        """
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
                # skip targets already in-flight to avoid duplicate measurements
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

        # SPREAD: bypass choose_arm(), allocate K across eligible arms.
        return self._queue_measurement_spread(batch_size)

    def absorb_measurement(self, result: dict[str, str]):
        """
        Measurements from queue are absorbed. Rewards are then propogated for the arm, the targets
        are disabled, QValue is updated.
        takes a dictionary of structure:
        {
            "target": str,
            "blocked": bool,
        }
        Returns a dictionary of structure:
        {
            "action": self._selected_arm_key,
            "target": self._selected_targets,
            "reward": round(measurement_result, 2),
            "q_value": round(observed_value, 2),
            "is_blocked": 1 if is_blocked else 0,
            "is_optimal": 1 if is_optimal else 0,
            "coverage": self.get_blocklist_coverage()
        }
        """
        t_name = result["target"]
        if t_name not in self._selected_targets.keys():
            raise KeyError(f"{self.country_name}: Received results from unknown target {t_name}")
        
        pending: Pending = self._selected_targets[t_name]

        is_blocked = result["blocked"]
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

        # call before observe
        is_optimal = self.is_optimal_action(arm_seq)
        observed_value = self.observe(arm_key, reward)

        assert observed_value is not None, "Missing observed value, expecting a float"
        pending.observed_value = observed_value

        self.propagate_rewards(arm_key)
        self.disable_target(pending.node_id)
        self.update_optimal_value()

        # set selected nodes explored
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

        del self._selected_targets[t_name]

        return return_value

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
