from models.ucb.ucb_naive import UCBNaive
import models.base.action_space as action_space_module
from typing import List
from Infrastructure.utils.structures import (
    BatchSizeMethod,
    BatchSelectionMethod,
    PropagationMethod
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
        self._selected_targets = {}
        self.selection_method = target_selection
        self.size_method = batch_size_method
        self.prop_method = qval_propagation_method

    def choose_targets(
        self, selected_arm_key: str, selected_arm_name: str, selection_size: int
    ) -> List[str]:
        def topk_from_arm():
            """Method: Choose top-k from chosen arm"""
            chosen_targets = self.action_space.sample_successors(
                selected_arm_key,
                n_samples=selection_size,
                use_rank_weights=self.sample_by_target_rank,
            )
            return chosen_targets[0:selection_size]
        
        def uniform_spread():
            """Method: Spread selection over all arms equally"""
            # not implemented
            return []
        
        def weighted_spread():
            """Method: Distribute k targets over arms depending on previous arm success"""
            # not implemented
            return []
        

        match self.selection_method:
            case BatchSelectionMethod.TOP_K_FROM_ARM:
                return topk_from_arm
            case BatchSelectionMethod.UNIFORM_SPREAD:
                return uniform_spread
            case BatchSelectionMethod.WEIGHTED_SPREAD:
                return weighted_spread

    def queue_measurement(self, batch_size) -> list[str]:
        """
        Selects an arm, then selects a group of targets from that arm.
        Returns a list of targets to measure.
        """
        self.selected_arm_seq = self.choose_arm()
        self._selected_arm_key = self.selected_arm_seq[-1]

        self._selected_arm_name = self.action_space.get(self._selected_arm_key)[
            action_space_module.NAME
        ]
        # print(f"selected arm: index: {selected_arm_index}, name: {selected_arm_name}")
        if self.verbose and self.logfile:
            self.logfile.write(str(self._selected_arm_name) + LOG_FILE_DELIMITER)

        new_targets = {
            self.action_space.get(t)[action_space_module.NAME]: {
                "node": t,
                "result": {
                    "blocked": False,
                    "reward": 0.0,
                },  # assume all not blocked by default
            }
            for t in self.choose_targets(
                self._selected_arm_key, self._selected_arm_name, batch_size
            )
        }

        self._selected_targets.update(new_targets)
        # for t_name in self._selected_targets.keys():
        #     self.disable_target(self._selected_targets[t_name]["node"])

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
        is_blocked = result["blocked"]
        measurement_result = 1 if is_blocked else 0
        self._selected_targets[t_name]["result"]["blocked"] = is_blocked
        self._selected_targets[t_name]["result"]["reward"] = measurement_result

        if is_blocked:
            self.update_blocklist_target_found(t_name)

        if self.verbose and self.logfile:
            self.logfile.write(
                t_name + LOG_FILE_DELIMITER + str(measurement_result) + "\n"
            )

        # call before observe
        is_optimal = self.is_optimal_action(self.selected_arm_seq)

        observed_value = self.observe(
            self._selected_arm_key, self._selected_targets[t_name]["result"]["reward"]
        )
        assert observed_value is not None, "Missing observed value, expecting a float"
        self._selected_targets[t_name]["observed_value"] = observed_value

        self.propagate_rewards(self._selected_arm_key)
        self.disable_target(self._selected_targets[t_name]["node"])
        self.update_optimal_value()

        # set selected nodes explored
        for n in self.selected_arm_seq + [self._selected_targets[t_name]["node"]]:
            self.action_space.get(n)[action_space_module.EXPLORED] = True

        return_value = {
            "action": self._selected_arm_key,
            "targets": t_name,
            "rewards": round(self._selected_targets[t_name]["result"]["reward"], 2),
            "q_value": round(self._selected_targets[t_name]["observed_value"], 2),
            "is_blocked": (
                1 if self._selected_targets[t_name]["result"]["blocked"] else 0
            ),
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
