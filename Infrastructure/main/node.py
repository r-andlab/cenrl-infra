from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, Callable
import os
import sys
from Infrastructure.models.BatchUCB import BatchUCB
from enum import Enum
from Infrastructure.utils.structures import MeasurementResponse
from dataclasses import asdict
import pandas as pd
from models.base.model import run_preprocessor
import models.base.action_space as action_space_module
from pathlib import Path


class NodeState(Enum):
    IDLE = 1
    READY = 2
    AWAITING_MEASUREMENTS = 3
    DONE = 4


class RegionalNode:
    def __init__(
        self,
        params: dict,
        country_name: str,
        vps: set[str],
        model_klass: BatchUCB,
        output_folder: str = None,
        batch_size: int = 10,
        action_space_folder: str = None,
        **kwargs: Any,
    ):
        self.params = params.copy()
        self.country_name_standard = country_name.replace(' ', '_')
        action_space_csv = None
        if action_space_folder:
            path = Path(action_space_folder)
            action_space_csv = path / self.country_name_standard / f"{self.country_name_standard}.csv"
            if not action_space_csv.exists():
                action_space_csv = None
            else:
                action_space_csv = str(action_space_csv)
        if output_folder:
            sections: list[str] = output_folder.split("/")
            sections[-1] = f"{country_name.replace(' ', '_')}/{sections[-1]}"
            self.params["outfile_csv"] = "/".join(sections)
            os.makedirs(os.path.dirname(self.params["outfile_csv"]), exist_ok=True)
        self.country: str = country_name
        self.active_vps: set[str] = set(vps)
        self.inactive_vps: set[str] = set()
        self.model: BatchUCB = model_klass(self.params, **kwargs)
        self.model.output_directory = os.path.dirname(self.params["outfile_csv"])
        self.model.outfile = Path(self.model.output_directory) / f"{self.country_name_standard}.csv"
        if action_space_csv:
            self.model.action_value_file = action_space_csv
        self.state: int = NodeState.IDLE
        self.batch_size = batch_size
        self.in_flight = 0
        self.model.current_epoch_num = 0
        self.episode_stats = []
        self.episode_all_stats = []
        self.episode_idx = 1
        self.active_measurements: List[MeasurementResponse] = []
        self.stat_df = None
        self.initialize()

    def set_measurements_per_episode(self, num_data: int):
        if (
            self.model.measurements_per_episode == "run_until_exhaustion"
            or self.model.measurements_per_episode > num_data
        ):
            self.model.measurements_per_episode = num_data
        assert self.model.measurements_per_episode <= num_data

    def initialize(
        self,
        action_space_klass: Callable = action_space_module.ActionSpaceBase,
        save_stats: bool = True,
    ) -> pd.DataFrame:

        num_data, action_space_df = run_preprocessor(
            self.model.action_space_file,
            self.model.features,
            self.model.consider_unknown,
        )

        self.set_measurements_per_episode(num_data)
        self.set_action_space(action_space_df, self.model.features, action_space_klass)
        if len(self.model.blocklist_unique_count) == 0:
            self.model.set_blocklist_unique_counts_based_on_action_space()
        self.save_stats = save_stats

    def set_action_space(
        self,
        action_space_df: pd.DataFrame,
        features: List[str],
        action_space_klass: Callable,
    ):
        initial_value_estimate = action_space_module.DEFAULT_Q_VALUE
        if hasattr(self.model, "initial_value_estimate"):
            initial_value_estimate = getattr(self.model, "initial_value_estimate")

        action_value_file = None
        if hasattr(self.model, "action_value_file") and getattr(self.model, "action_value_file"):
            # print(f"Found action value file! {getattr(self.model, 'action_value_file')}")
            action_value_file = getattr(self.model, "action_value_file")

        # build the action space
        self.model.action_space = action_space_klass(
            self.model.output_directory,
            action_space_df,
            features,
            self.model.target_feature,
            default_q_value=initial_value_estimate,
            multiple_parents=self.model.action_space_multi_parents,
            action_value_file=action_value_file,
        )

    def write_stats(self):
        print(f"{self.country}: Episode Process {os.getpid()} - Saving stats and model")
        self.model.save()
        self.stat_df.to_csv(self.model.outfile, index=False)
        return

    def maybe_request_more(self) -> Optional[List[str]]:
        if self.state is NodeState.IDLE:
            self.state = NodeState.READY
        if self.state is NodeState.READY:
            # print("Trying to step!")
            if self.model.can_step() and self.in_flight < self.batch_size and self.model.current_epoch_num < self.model.measurements_per_episode:
                request_size = max(
                    min(
                        self.batch_size - self.in_flight,
                        self.model.measurements_per_episode - (self.model.current_epoch_num + self.in_flight)
                        ),
                    0
                )
                if request_size > 0:
                    print(f"request_size: {request_size}, batch_size: {self.batch_size}, in_flight: {self.in_flight}, measurements_per_ep: {self.model.measurements_per_episode}, epoch_num: {self.model.current_epoch_num}\n\n")
                    targets = self.model.queue_measurement(request_size)
                    print(f"{self.country_name_standard} requesting {targets}")
                    self.in_flight += len(targets)
                    # self.state = NodeState.AWAITING_MEASUREMENTS
                    return targets
                return None
            else:
                if not self.model.can_step():
                    print("Can't Step!")
        return None

    def maybe_update_model(
        self, measurements: list[MeasurementResponse], save_stats: bool = True
    ):
        if len(measurements) > 0:
            self.in_flight -= len(measurements)
            # self.active_measurements += measurements

            for m in measurements:
                episode_stat = {
                    "episode": self.episode_idx,
                    "time": self.model.current_epoch_num + 1,
                }
                episode_stat.update(
                    self.model.absorb_measurement(asdict(m))
                )
                self.episode_stats.append(episode_stat)

            if (
                self.model.current_epoch_num != 0
                and self.model.current_epoch_num % 100 == 0
            ):
                print(
                    f"{self.country}: Episode Process {os.getpid()} - Done with {self.model.current_epoch_num} iterations"
                )
            # self.active_measurements = []
            # self.state = NodeState.READY
            # print("Done epoch, setting state back to READY")
            # end of epoch
            if (
                self.model.current_epoch_num
                < self.model.measurements_per_episode - 1
            ):
                self.model.current_epoch_num += len(measurements)
            if self.episode_idx <= self.model.num_episodes and self.in_flight == 0:
                self.episode_all_stats += self.episode_stats
                if self.episode_idx < self.model.num_episodes:
                    self.model.reset()
                    self.episode_idx += 1
                else:
                    print("Done")
                    self.state = NodeState.DONE
                    self.stat_df = pd.DataFrame(
                        columns=list(self.episode_all_stats[0].keys())
                    )
                    self.stat_df = self.stat_df.from_dict(self.episode_all_stats)
                    if save_stats:
                        self.write_stats()
                    return self.stat_df

        return

    def get_vps(self):
        return self.active_vps

    def add_vp(self, vp):
        self.active_vps.add(vp)

    def deactivate_vp(self, vp):
        if vp in self.active_vps:
            self.active_vps.remove(vp)
        self.inactive_vps.add(vp)

    def delete_vp(self, vp):
        if vp in self.active_vps:
            self.active_vps.remove(vp)
        if vp in self.inactive_vps:
            self.inactive_vps.remove(vp)
