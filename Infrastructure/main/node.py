from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional, Callable, Set
import logging
import os
import sys
from Infrastructure.models.BatchUCB import BatchUCB
from Infrastructure.utils.structures import MeasurementResponse, BatchSelectionMethod, BatchSizeMethod, PropagationMethod, NodeState
from dataclasses import asdict
import pandas as pd
from models.base.model import run_preprocessor
import models.base.action_space as action_space_module
from pathlib import Path

logger = logging.getLogger(__name__)


class RegionalNode:
    def __init__(
        self,
        params: Dict,
        country_name: str,
        model_klass: BatchUCB,
        output_folder: str = None,
        batch_size: int = 10,
        action_space_folder: str = None,
        **kwargs: Any,
    ):
        self.params = params.copy()
        self.country_name_standard = country_name.replace(' ', '_')
        self.output_folder = output_folder
        action_space_csv = None
        if action_space_folder:
            path = Path(action_space_folder)
            action_space_csv = path / self.country_name_standard / f"{self.country_name_standard}.csv"
            if not action_space_csv.exists():
                action_space_csv = None
            else:
                action_space_csv = str(action_space_csv)
        if output_folder:
            base_path = Path(output_folder)  # e.g. outputs/outtest9
            country_dir = base_path / country_name.replace(" ", "_")
            csv = country_dir / f"{country_name.replace(' ', '_')}.csv"
            # print(base_path, country_dir, csv)
            # Create the directory
            country_dir.mkdir(parents=True, exist_ok=True)

            # If you want outfile_csv to point inside that folder:
            self.params["outfile_csv"] = str(csv)
        self.country: str = country_name
        self.model: BatchUCB = model_klass(self.params, country_name, **kwargs)
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
        logger.info(f"Creating action space for country {self.country_name_standard}")
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
        if self.stat_df is not None:
            self.stat_df.to_csv(self.model.outfile, index=False)
        elif self.episode_all_stats or self.episode_stats:
            all_stats = self.episode_all_stats + self.episode_stats
            if all_stats:
                pd.DataFrame(all_stats).to_csv(self.model.outfile, index=False)
        return

    def soft_reset(self) -> None:
        """Daily soft reset: re-enable all sleeping targets, reset epoch counter.
        Does NOT call model.reset() — Q-values and arm counts are preserved (D-03).
        """
        logger.info("%s: performing daily soft reset", self.country)
        self.model.action_space.wake_up_all_nodes()
        self.model.current_epoch_num = 0
        self.in_flight = 0
        self.state = NodeState.IDLE
        self.episode_stats = []

    def _remaining_capacity(self) -> int:
        return max(self.batch_size - self.in_flight, 0)

    def _remaining_steps_this_episode(self) -> int:
        # how many *more* measurements we still need to COMPLETE this episode
        return max(self.model.measurements_per_episode - self.model.current_epoch_num, 0)

    def _remaining_steps_including_inflight(self) -> int:
        # how many more we can *request* without exceeding measurements_per_episode
        return max(self.model.measurements_per_episode - (self.model.current_epoch_num + self.in_flight), 0)

    def maybe_request_more(self) -> Optional[List[str]]:
        if self.state is NodeState.IDLE:
            self.state = NodeState.READY
        if self.state is not NodeState.READY:
            return None

        if not self.model.can_step():
            return None

        requestable = min(self._remaining_capacity(), self._remaining_steps_including_inflight())
        if requestable <= 0:
            return None

        targets = self.model.queue_measurement(requestable)
        if not targets:
            return None

        self.in_flight += len(targets)
        return targets

    def _append_measurements(self, measurements: List[MeasurementResponse]) -> int:
        """
        Absorb batch of responses. Assign monotonically increasing step indexes within an episode.

        :param self: Node responsible for current model
        :param measurements: List of incoming measurement responses to absorb into model
        :type measurements: List[MeasurementResponse]
        :return: Number of measurements successfully absorbed
        """
        absorbed = 0
        start_step = self.model.current_epoch_num
        for m in measurements:
            try:
                result = self.model.absorb_measurement(asdict(m))
            except KeyError:
                logger.warning(
                    "%s: skipping duplicate/stale result for target %s",
                    self.country, m.target,
                )
                continue
            step_idx = start_step + absorbed
            episode_stat = {
                "episode": self.episode_idx,
                "time": step_idx + 1,
            }
            episode_stat.update(result)
            self.episode_stats.append(episode_stat)
            absorbed += 1

        self.model.current_epoch_num += absorbed
        return absorbed

    def _finish_episode_if_ready(self, save_stats: bool) -> Optional[pd.DataFrame]:
        """If daily measurement quota reached and no in-flight, idle until reset.

        Under continuous operation (D-06/D-07), one calendar day = one episode.
        Reaching the quota means idle-until-midnight, NOT termination.
        """
        if self.in_flight != 0:
            return None

        if self.model.current_epoch_num < self.model.measurements_per_episode:
            return None

        # Daily quota reached — idle until next midnight reset
        logger.info(
            "%s: daily quota of %d measurements reached, idling until reset",
            self.country, self.model.measurements_per_episode,
        )
        if self.episode_stats:
            self.episode_all_stats += self.episode_stats
            self.episode_stats = []
        self.state = NodeState.IDLE
        # Do NOT set NodeState.DONE — do NOT call model.reset()
        return None

    def maybe_update_model(
        self, measurements: list[MeasurementResponse], save_stats: bool = True
    ):
        """
        Update models with measurement responses

        :param self: Node responsible for model
        :param measurements: List of incoming measurement responses to absorb into model
        :type measurements: list[MeasurementResponse]
        :param save_stats: flag that determines whether output is written
        :type save_stats: bool
        """
        if not measurements:
            return None

        absorbed = self._append_measurements(measurements)

        self.in_flight -= absorbed
        if self.in_flight < 0:
            self.in_flight = 0

        if self.model.current_epoch_num != 0 and self.model.current_epoch_num % 100 == 0:
            print(f"{self.country}: Episode Process {os.getpid()} - Done with {self.model.current_epoch_num} iterations")

        return self._finish_episode_if_ready(save_stats)

