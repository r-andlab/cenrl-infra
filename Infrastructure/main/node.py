from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, Callable, Set
import csv
import json
import logging
import os
import sys
import tempfile
import time
from Infrastructure.models.BatchUCB import BatchUCB
from Infrastructure.utils.persistence import find_latest_state, load_graph_and_sidecar, validate_sidecar
from Infrastructure.utils.structures import MeasurementResponse, BatchSelectionMethod, BatchSizeMethod, PropagationMethod, NodeState
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
        on_reset: Optional[Callable[[str], None]] = None,
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
            csv_path = country_dir / f"{country_name.replace(' ', '_')}.csv"
            # print(base_path, country_dir, csv_path)
            # Create the directory
            country_dir.mkdir(parents=True, exist_ok=True)

            # If you want outfile_csv to point inside that folder:
            self.params["outfile_csv"] = str(csv_path)

            # Phase 3 D-01: per-country flush-on-write measurements CSV path.
            # File is opened lazily on first row append in _append_measurements
            # (D-15: "Files are created on first row").
            measurements_path = country_dir / f"{country_name.replace(' ', '_')}_measurements.csv"
            self._measurements_path: Optional[Path] = measurements_path
            self._measurements_file = None
            self._measurements_writer: Optional[csv.DictWriter] = None
            self._measurements_header_written: bool = False
        else:
            self._measurements_path: Optional[Path] = None
            self._measurements_file = None
            self._measurements_writer: Optional[csv.DictWriter] = None
            self._measurements_header_written: bool = False
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
        # Phase 02.1 D-12: optional orchestrator-side cleanup callback fired
        # after each per-country soft_reset inside _finish_episode_if_ready.
        self._on_reset: Optional[Callable[[str], None]] = on_reset
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

        # Attempt to load saved state (D-09: auto-detect on startup). The fresh
        # action space built above remains in place; load_checkpoint() patches
        # Q-values / arm counts onto it from any saved snapshot.
        state_dir = Path(self.model.output_directory) / "state"
        if state_dir.exists():
            self.load_checkpoint(state_dir)

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
        """Write per-iteration snapshot CSV and append to rolling per-country CSV (D-05, D-07).

        Caller (_finish_episode_if_ready) is responsible for rotating
        episode_stats → episode_all_stats BEFORE invoking this method, so that
        episode_all_stats holds all rows for the iteration about to be written.

        Files written (per D-07):
        - <country>.csv               (rolling; appended on each iteration; header on first write only)
        - <country>_iter_NNN.csv      (immutable per-iteration snapshot; full overwrite each call)

        After writing, episode_all_stats is cleared so the next iteration starts
        with an empty rolling buffer and the rolling CSV is not double-appended.
        """
        print(f"{self.country}: Episode Process {os.getpid()} - Saving stats and model")
        self.model.save()

        # Legacy stat_df path: external test harnesses still set this directly. Preserve
        # the original overwrite behavior for compatibility — D-05 only governs the
        # episode_stats path used by the live infrastructure run.
        if self.stat_df is not None:
            self.stat_df.to_csv(self.model.outfile, index=False)
            return

        rows = list(self.episode_all_stats) + list(self.episode_stats)
        if not rows:
            return

        rolling_path = Path(self.model.outfile)
        snapshot_path = rolling_path.with_name(
            f"{rolling_path.stem}_iter_{self.episode_idx:03d}{rolling_path.suffix}"
        )

        # Per-iteration snapshot — full overwrite, immutable artifact.
        pd.DataFrame(rows).to_csv(snapshot_path, index=False)

        # Rolling per-country CSV — append-mode. Header on first write only.
        header_needed = not rolling_path.exists()
        pd.DataFrame(rows).to_csv(
            rolling_path, mode="a", header=header_needed, index=False,
        )

        # Clear the rolled-over buffer so the next iteration's rotate-then-write
        # sequence does not double-append rows we have already persisted.
        self.episode_all_stats = []
        return

    def _write_sidecar(self, json_path: Path) -> None:
        """Write JSON sidecar with UCB scalars, run config snapshot, and metadata.

        Per D-14: contains UCB counters (exploration_epoch_num, current_epoch_num),
        daily progress (measurements_completed_today), run config snapshot
        (c, step_size, initial_value_estimate, strategy enums as .name), and
        metadata (save_timestamp, country_code). Atomic write via .tmp + os.replace
        to prevent corrupted sidecars on crash mid-write (T-02-02 mitigation).
        """
        sidecar = {
            # UCB counters
            "exploration_epoch_num": self.model.exploration_epoch_num,
            "current_epoch_num": self.model.current_epoch_num,
            # Daily progress
            "measurements_completed_today": self.model.current_epoch_num,
            # Run config snapshot
            "c": self.model.c,
            "step_size": self.model.step_size,
            "initial_value_estimate": self.model.initial_value_estimate,
            "selection_method": self.model.selection_method.name,
            "size_method": self.model.size_method.name,
            "prop_method": self.model.prop_method.name,
            # Metadata
            "save_timestamp": datetime.now(timezone.utc).isoformat(),
            "country_code": self.country,
        }
        # WR-08 fix: use NamedTemporaryFile (delete=False) for unique tmp
        # names per process. The previous fixed `<path>.tmp` was vulnerable
        # to two concurrent writers racing on the same tmp file and producing
        # a truncated final json.
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(json_path.parent),
                prefix=json_path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(sidecar, f, indent=2)
            os.replace(tmp_path, str(json_path))
        except Exception:
            # Best-effort cleanup of the unique tmp file if replace failed.
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

    def save_checkpoint(self, state_dir: Path, prefix: str = "checkpoint") -> None:
        """Save rolling checkpoint (overwrites fixed filenames per D-02).

        Writes <state_dir>/<prefix>.graphml via ActionSpaceBase.checkpoint_save()
        and <state_dir>/<prefix>.json via _write_sidecar(). Errors are logged
        but never raised — losing one checkpoint is better than crashing the
        orchestrator (D-12).
        """
        state_dir.mkdir(parents=True, exist_ok=True)
        graphml_path = state_dir / f"{prefix}.graphml"
        json_path = state_dir / f"{prefix}.json"
        try:
            self.model.action_space.checkpoint_save(str(graphml_path))
            self._write_sidecar(json_path)
        except Exception as exc:
            logger.error("%s: save_checkpoint failed: %s", self.country, exc)

    def save_daily(self, date_str: str, state_dir: Path) -> None:
        """Save date-stamped daily boundary snapshot (per D-06).

        Must be called BEFORE soft_reset() — soft_reset() clears
        current_epoch_num and SLEEPING attributes (Pitfall 4).
        """
        state_dir.mkdir(parents=True, exist_ok=True)
        graphml_path = state_dir / f"action_space_{date_str}.graphml"
        json_path = state_dir / f"state_{date_str}.json"
        try:
            self.model.action_space.checkpoint_save(str(graphml_path))
            self._write_sidecar(json_path)
            logger.info("%s: daily state saved to %s", self.country, state_dir)
        except Exception as exc:
            logger.error("%s: save_daily failed: %s", self.country, exc)

    def save_iteration(self, idx: int, state_dir: Path) -> None:
        """Save per-iteration state snapshot under state/iterations/ (per Phase 02.1 D-06, D-07).

        Must be called BEFORE soft_reset() — soft_reset() clears
        current_epoch_num and SLEEPING attributes (Pitfall 4 / D-08).

        Files written (under <state_dir>/iterations/):
        - iter_NNN.graphml — action space snapshot for iteration NNN (zero-padded to 3 digits)
        - iter_NNN.json    — JSON sidecar with UCB scalars + run config

        Atomic writes via checkpoint_save (.tmp + os.replace) and _write_sidecar (.tmp + os.replace).
        Per-country fault isolation is the caller's responsibility (try/except in
        _finish_episode_if_ready); this method itself does not swallow exceptions.
        """
        iter_state_dir = state_dir / "iterations"
        iter_state_dir.mkdir(parents=True, exist_ok=True)
        graphml_path = iter_state_dir / f"iter_{idx:03d}.graphml"
        json_path = iter_state_dir / f"iter_{idx:03d}.json"
        try:
            self.model.action_space.checkpoint_save(str(graphml_path))
            self._write_sidecar(json_path)
            logger.info(
                "%s: iteration %d state saved to %s",
                self.country, idx, iter_state_dir,
            )
        except Exception as exc:
            logger.error("%s: save_iteration failed: %s", self.country, exc)
            raise

    def load_checkpoint(self, state_dir: Path) -> bool:
        """Load saved state from state_dir and merge into the current action space.

        Returns True iff a valid (graphml, json) pair was found, parsed, and
        merged. Returns False on missing state, corrupt files (D-12), or
        sidecar schema validation failure (T-02-04 mitigation).

        Graph merge (D-10): the current action space was built fresh from CSV
        before this call, so its node set reflects the latest input. We patch
        Q-values, ACTION_ATTEMPTS, SLEEPING, and EXPLORED from the saved
        graph onto matching nodes of the live graph. Targets present only in
        the saved graph are silently dropped (removed from the CSV). Targets
        only in the current graph keep their default values from build_graph().
        """
        graphml_path, json_path = find_latest_state(state_dir)
        if graphml_path is None or json_path is None:
            return False

        saved_graph, sidecar = load_graph_and_sidecar(graphml_path, json_path)
        if saved_graph is None or sidecar is None:
            return False

        if not validate_sidecar(sidecar):
            return False

        # D-10: merge Q-values from saved graph onto the freshly built graph.
        live_graph = self.model.action_space._graph
        merged_count = 0
        for node_id in saved_graph.nodes():
            if not live_graph.has_node(node_id):
                # Node was in the saved snapshot but not in the current CSV — drop it.
                continue
            saved_attrs = saved_graph.nodes[node_id]
            live_attrs = live_graph.nodes[node_id]
            for attr in (
                action_space_module.Q_VALUE,
                action_space_module.ACTION_ATTEMPTS,
                action_space_module.SLEEPING,
                action_space_module.EXPLORED,
            ):
                if attr in saved_attrs:
                    live_attrs[attr] = saved_attrs[attr]
            merged_count += 1

        # Restore UCB scalars
        self.model.exploration_epoch_num = sidecar["exploration_epoch_num"]
        self.model.current_epoch_num = sidecar["current_epoch_num"]

        # Hyperparameter mismatch check — log warning but proceed
        for hp in ("c", "step_size", "initial_value_estimate"):
            saved_val = sidecar.get(hp)
            current_val = getattr(self.model, hp, None)
            if saved_val != current_val:
                logger.warning(
                    "%s: hyperparameter mismatch: saved %s=%r vs current %s=%r",
                    self.country, hp, saved_val, hp, current_val,
                )

        # Restore enum fields from sidecar string names; tolerate invalid names
        # by logging a warning and keeping the current model value (T-02-04).
        try:
            self.model.selection_method = BatchSelectionMethod[sidecar["selection_method"]]
        except KeyError:
            logger.warning(
                "%s: unknown enum name for selection_method in sidecar — keeping current value",
                self.country,
            )
        try:
            self.model.size_method = BatchSizeMethod[sidecar["size_method"]]
        except KeyError:
            logger.warning(
                "%s: unknown enum name for size_method in sidecar — keeping current value",
                self.country,
            )
        try:
            self.model.prop_method = PropagationMethod[sidecar["prop_method"]]
        except KeyError:
            logger.warning(
                "%s: unknown enum name for prop_method in sidecar — keeping current value",
                self.country,
            )

        logger.info(
            "%s: loaded checkpoint from %s (%d nodes merged, exploration_epoch=%d)",
            self.country,
            graphml_path,
            merged_count,
            self.model.exploration_epoch_num,
        )
        return True

    def soft_reset(self) -> None:
        """Per-country iteration soft reset: re-enable all sleeping targets, reset epoch
        counter, and bump the per-country iteration counter (Phase 02.1 D-04).
        Does NOT call model.reset() — Q-values and arm counts are preserved (Phase 01 D-03).
        """
        logger.info("%s: performing iteration soft reset (idx=%d)", self.country, self.episode_idx)
        self.model.action_space.wake_up_all_nodes()
        self.model.current_epoch_num = 0
        self.in_flight = 0
        self.state = NodeState.IDLE
        self.episode_stats = []
        self.episode_idx += 1

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
        :return: Number of in-flight slots this batch reclaims. Equal to the
            count of measurements successfully absorbed by the model PLUS any
            duplicate/stale results that were skipped (WR-02 fix). The caller
            decrements ``self.in_flight`` by this value so accounting stays
            symmetric with scheduling, which always increments ``in_flight``
            regardless of whether the result later turns out to be a duplicate.
            ``current_epoch_num`` is only advanced by the count of measurements
            that were actually absorbed (not by skipped duplicates), so the
            episode quota tracks real progress.
        """
        absorbed = 0
        consumed = 0
        start_step = self.model.current_epoch_num
        for m in measurements:
            try:
                result = self.model.absorb_measurement(asdict(m))
            except KeyError:
                logger.warning(
                    "%s: skipping duplicate/stale result for target %s",
                    self.country, m.target,
                )
                # WR-02 fix: a duplicate/stale result still consumed an
                # in-flight slot when it was originally scheduled. Count it
                # toward consumed so the caller's `in_flight -= consumed`
                # decrement does not under-count, which would leave
                # `in_flight` permanently above zero and block
                # _finish_episode_if_ready.
                consumed += 1
                continue
            step_idx = start_step + absorbed
            episode_stat = {
                "episode": self.episode_idx,
                "time": step_idx + 1,
            }
            episode_stat.update(result)

            # Phase 3 D-02: per-measurement CSV row (12 columns, exact order).
            utc_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            latency_ms: Optional[float] = None
            if m.scheduled_at_monotonic is not None:
                latency_ms = round(
                    (time.monotonic() - m.scheduled_at_monotonic) * 1000.0, 1
                )
            meas_row = {
                "target":        result.get("targets"),  # BatchUCB return uses plural key
                "arm":           result.get("action"),
                "blocked":       result.get("is_blocked"),
                "reward":        result.get("rewards"),
                "q_value":       result.get("q_value"),
                "utc_timestamp": utc_ts,
                "episode_idx":   self.episode_idx,
                "step_idx":      step_idx + 1,
                "is_optimal":    result.get("is_optimal"),
                "coverage":      result.get("coverage"),
                "latency_ms":    latency_ms,
                "vp_count":      m.vp_count,
            }
            self._write_measurements_row(meas_row)

            self.episode_stats.append(episode_stat)
            absorbed += 1
            consumed += 1

        self.model.current_epoch_num += absorbed
        return consumed

    def _write_measurements_row(self, row: Dict[str, Any]) -> None:
        """Append one row to <country>_measurements.csv (Phase 3 D-01..D-06).

        Lazy file open on first call (D-15). Writes the header exactly once
        (D-03). Flushes after every row so partial runs are readable in
        pandas/R for thesis analysis (OBSV-01). Per-country fault isolation:
        a write failure is logged but never raised — the orchestrator tick
        and other countries continue (Phase 02.1 D-04 invariant).
        """
        if self._measurements_path is None:
            return  # No output_folder configured — silent drop, matches _configure_logging fallback.
        try:
            if self._measurements_file is None:
                self._measurements_file = open(
                    self._measurements_path, "a", newline=""
                )
                self._measurements_writer = csv.DictWriter(
                    self._measurements_file,
                    fieldnames=[
                        "target", "arm", "blocked", "reward", "q_value",
                        "utc_timestamp", "episode_idx", "step_idx",
                        "is_optimal", "coverage", "latency_ms", "vp_count",
                    ],
                )
                # Header iff the file is empty (handles both fresh creation
                # and appending to a previous run's truncated file).
                if self._measurements_path.stat().st_size == 0:
                    self._measurements_writer.writeheader()
                self._measurements_header_written = True
            self._measurements_writer.writerow(row)
            self._measurements_file.flush()
        except Exception as exc:
            logger.error(
                "%s: measurements row write failed: %s",
                self.country, exc,
            )

    def close_measurements(self) -> None:
        """Flush + close the measurements CSV file handle (D-15 shutdown path).

        Idempotent. MUST NOT be called from soft_reset or _on_country_reset
        (writer is intentionally long-lived across iteration boundaries per
        Phase 3 D-01).
        """
        if self._measurements_file is None:
            return
        try:
            self._measurements_file.flush()
            self._measurements_file.close()
        except Exception as exc:
            logger.error(
                "%s: failed to close measurements writer: %s",
                self.country, exc,
            )
        finally:
            self._measurements_file = None
            self._measurements_writer = None

    def _finish_episode_if_ready(self, save_stats: bool) -> Optional[pd.DataFrame]:
        """Per-country iteration boundary: when measurement quota is reached and no
        in-flight measurements remain, persist outputs then soft-reset (Phase 02.1 D-02, D-03, D-08).

        Existing guards (preserved):
        - in_flight != 0  → drain in-progress; return None and wait (D-03)
        - current_epoch_num < measurements_per_episode → quota not yet hit; return None

        On trigger (current_epoch_num >= measurements_per_episode AND in_flight == 0):
        Strict ordering per D-08: write_stats(CSV) → save_iteration(state) → soft_reset().
        Both saves MUST complete before soft_reset() zeroes current_epoch_num and clears
        SLEEPING via wake_up_all_nodes() (Pitfall 4 from Phase 02 save_daily docstring).

        Per-country fault isolation (Phase 02 D-04 invariant): each save is wrapped in
        try/except so a failing save does not block the soft_reset path. Soft_reset
        always runs.
        """
        if self.in_flight != 0:
            return None

        if self.model.current_epoch_num < self.model.measurements_per_episode:
            return None

        logger.info(
            "%s: iteration %d quota of %d measurements reached, performing per-country reset",
            self.country, self.episode_idx, self.model.measurements_per_episode,
        )

        # Rotate this iteration's rows into the rolling buffer so write_stats sees them
        # (D-05: rolling CSV append uses episode_all_stats; per-iteration snapshot uses
        # the same rows).
        if self.episode_stats:
            self.episode_all_stats += self.episode_stats
            self.episode_stats = []

        # Resolve state_dir via the same convention as Orchestrator._state_dir_for —
        # node has self.model.output_directory == "<output>/<country>/" already.
        state_dir = Path(self.model.output_directory) / "state"

        # Per-country fault isolation: D-04 invariant means a failed save MUST NOT
        # prevent soft_reset from running for this country.
        try:
            self.write_stats()
        except Exception as exc:
            logger.error(
                "%s: write_stats failed at iteration %d boundary: %s",
                self.country, self.episode_idx, exc,
            )

        try:
            self.save_iteration(self.episode_idx, state_dir)
        except Exception as exc:
            logger.error(
                "%s: save_iteration failed at iteration %d: %s",
                self.country, self.episode_idx, exc,
            )

        # Always soft-reset, even if a save raised. Soft_reset bumps episode_idx (D-04).
        self.soft_reset()

        # Per-country cleanup callback (Phase 02.1 D-12). The orchestrator clears its
        # _inflight_times[country] and _last_delay_warn[country] entries here so the
        # next iteration starts with a clean delayed-result slate. Per-country fault
        # isolation: callback errors are logged, not re-raised.
        if self._on_reset is not None:
            try:
                self._on_reset(self.country)
            except Exception as exc:
                logger.error(
                    "%s: on_reset callback failed: %s",
                    self.country, exc,
                )

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

        # WR-02 fix: _append_measurements returns the number of in-flight
        # slots reclaimed (absorbed + skipped duplicates). Decrementing
        # in_flight by this value keeps accounting symmetric with the
        # schedule path that always increments in_flight on send,
        # regardless of whether the result later turns out to be a duplicate.
        consumed = self._append_measurements(measurements)

        self.in_flight -= consumed
        if self.in_flight < 0:
            logger.warning(
                "%s: in_flight went negative (%d), clamping to 0 — possible double-absorption",
                self.country, self.in_flight,
            )
            self.in_flight = 0

        if self.model.current_epoch_num != 0 and self.model.current_epoch_num % 100 == 0:
            print(f"{self.country}: Episode Process {os.getpid()} - Done with {self.model.current_epoch_num} iterations")

        return self._finish_episode_if_ready(save_stats)

