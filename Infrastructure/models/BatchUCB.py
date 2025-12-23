from models.ucb.ucb_naive import UCBNaive
import models.base.action_space as action_space_module
import typing
from common.utils import (
    LOG_FILE_DELIMITER,
    NO_DATE_BLOCKLIST,
    TARGET_FEATURE__DOMAIN,
    UNKNOWN_EMPTY,
    TARGET_FEATURE__SERVICE_IP,
    SERVER_FEATURE_ORDER,
)


class BatchUCB(UCBNaive):
    def choose_targets(self, selected_arm_key: str, selected_arm_name: str, selection_size: int) -> str:
        chosen_targets = self.action_space.sample_successors(
            selected_arm_key, n_samples=selection_size, use_rank_weights=self.sample_by_target_rank
        )
        return chosen_targets[0:selection_size]

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

        self._selected_targets = {
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

        if self.verbose and self.logfile:
            self.logfile.write(f"Selection Size {batch_size}" + LOG_FILE_DELIMITER)

        return list(self._selected_targets.keys())

    def absorb_measurement(self, results: list[dict[str, str]]):
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
        for result in results:
            t = result["target"]
            is_blocked = result["blocked"]
            measurement_result = 1 if is_blocked else 0
            self._selected_targets[t]["result"]["blocked"] = is_blocked
            self._selected_targets[t]["result"]["reward"] = measurement_result

            if is_blocked:
                self.update_blocklist_target_found(t)

            if self.verbose and self.logfile:
                self.logfile.write(
                    str(self._selected_targets[t])
                    + LOG_FILE_DELIMITER
                    + str(measurement_result)
                    + "\n"
                )

        # call before observe
        is_optimal = self.is_optimal_action(self.selected_arm_seq)

        for t in self._selected_targets.keys():
            observed_value = self.observe(self._selected_arm_key, self._selected_targets[t]["result"]["reward"])
            assert observed_value is not None, "Missing observed value, expecting a float"
            self._selected_targets[t]["observed_value"] = observed_value

        self.propagate_rewards(self._selected_arm_key)
        self.disable_target(self._selected_targets[t]["node"])
        self.update_optimal_value()

        # set selected nodes explored
        for n in (self.selected_arm_seq + [self._selected_targets[t]["node"] for t in self._selected_targets.keys()]):
            self.action_space.get(n)[action_space_module.EXPLORED] = True

        return {
            "action": self._selected_arm_key,
            "targets": list(self._selected_targets.keys()),
            "rewards": [
                round(self._selected_targets[t]["result"]["reward"], 2)
                for t in self._selected_targets.keys()
            ],
            "q_values": [
                round(self._selected_targets[t]["observed_value"], 2)
                for t in self._selected_targets.keys()
            ],
            "is_blocked": [
                1 if self._selected_targets[t]["result"]["blocked"] else 0
                for t in self._selected_targets.keys()
            ],
            "is_optimal": 1 if is_optimal else 0,
            "coverage": self.get_blocklist_coverage(),
        }

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
