from models.ucb.ucb_naive import UCBNaive
import models.base.action_space as action_space_module
from typing import List, Dict
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

    def choose_targets(
        self, selected_arm_key: str, selected_arm_name: str, selection_size: int = 10
    ) -> List[str]:
        if self.selection_method is BatchSelectionMethod.TOP_K_FROM_ARM:
            """Method: Choose top-k from chosen arm"""
            chosen_targets = self.action_space.sample_successors(
                selected_arm_key,
                n_samples=selection_size,
                use_rank_weights=self.sample_by_target_rank,
            )
            return chosen_targets[0:selection_size]
        elif self.selection_method is BatchSelectionMethod.UNIFORM_SPREAD:
            """Method: Spread selection over all arms equally"""
            raise NotImplementedError
        elif self.selection_method is BatchSelectionMethod.WEIGHTED_SPREAD:
            """Method: Distribute k targets over arms depending on previous arm success"""
            raise NotImplementedError

    def queue_measurement(self, batch_size) -> list[str]:
        """
        Selects an arm, then selects a group of targets from that arm.
        Returns a list of targets to measure.
        """
        arm_seq = self.choose_arm()
        arm_key = arm_seq[-1]
        arm_name = self.action_space.get(arm_key)[action_space_module.NAME]

        if self.verbose and self.logfile:
            self.logfile.write(str(arm_name) + LOG_FILE_DELIMITER)

        node_ids = self.choose_targets(arm_key, arm_name, batch_size)

        new_targets: Dict[str, Pending] = {}
        for node_id in node_ids:
            t_name = self.action_space.get(node_id)[action_space_module.NAME]
            # guard against NAME collisions
            if t_name in self._selected_targets or t_name in new_targets:
                # prefer unique key; for now, disambiguate, may be pointless
                t_name = f"{t_name}__{node_id}"

            new_targets[t_name] = Pending(
                node_id=node_id,
                arm_key=arm_key,
                arm_seq=arm_seq,
            )

        self._selected_targets.update(new_targets)
        
        if self.verbose and self.logfile:
            self.logfile.write(f"Selection Size {batch_size}" + LOG_FILE_DELIMITER)

        return list(new_targets.keys())

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
            raise KeyError(f"Received results from unknown target {t_name}")
        
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
