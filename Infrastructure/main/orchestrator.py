import csv
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta, time as dtime
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Set, Tuple

from models.ucb.ucb_naive import UCBNaiveParserOptions
from Infrastructure.apis.funneler import HyperQuackAPI
from Infrastructure.models.BatchUCB import BatchUCB
from Infrastructure.main.node import RegionalNode
from Infrastructure.main.vantage_points import VantagePoints
from Infrastructure.utils.eval_store import EvalStore
from Infrastructure.utils.structures import NodeState, TestPayload, Tag

logger = logging.getLogger(__name__)

# A VP is removed from service after this many controls_failed results.
VP_FAILURE_THRESHOLD = 5


class Orchestrator:
    def __init__(
        self,
        params: Dict,
        vantage_points: VantagePoints,
        go_api_endpoint: str,
        services: List[str],
        countries: List[str] = None,
        vps_per_country: int = 3,
        previous_values_folder: str = None,
        debug: bool = False,
        vp_pool_config: Optional[Dict] = None,
    ):
        self.params = params
        self.output_folder = params.get("output_directory", None)
        self.previous_values_folder = previous_values_folder
        if previous_values_folder:
            path = Path(previous_values_folder)
            if not path.exists() or not path.is_dir():
                self.previous_values_folder = None
        # WR-04 fix: removed redundant `os.makedirs(os.path.dirname(...))`
        # call. It crashed with FileNotFoundError on bare relative paths
        # (where dirname returns "") and was redundant anyway —
        # _configure_logging() below creates self.output_folder directly,
        # and per-country directories are created in RegionalNode.__init__.

        # Configure logging now that output_folder is known and the directory
        # exists. Must run BEFORE _bootstrap_vps() emits any log records.
        self._configure_logging()

        # Capture constructor args for run_config snapshot (Phase 3 D-13).
        self._go_api_endpoint: str = go_api_endpoint
        self._vp_pool_config: Optional[Dict] = vp_pool_config

        self.vantage_points = vantage_points
        self.services = services
        self.vps_per_country = vps_per_country

        # Shared eval store for VP evaluation results
        self.eval_store = EvalStore()

        # Create API with shared eval store
        self.api = HyperQuackAPI(
            go_api_endpoint,
            eval_store=self.eval_store,
            vantage_points=vantage_points,
            debug=debug,
        )

        # Nodes are created lazily once a VP passes evaluation.
        self.agents: Dict[str, RegionalNode] = {}
        self.completed_agents: set = set()

        # VP health tracking: (country, vp) -> consecutive failure count
        self._vp_failure_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        # Determine target countries
        if countries is None:
            countries = self.vantage_points.countries()
        self.target_countries = set(countries)

        # Add tags for each country
        for country in self.target_countries:
            self.api.add_tags(
                [
                    Tag(
                        tag=country,
                        result_output_file=f"{country}_result.jsonl",
                        eval_output_file=f"{country}_eval.jsonl",
                    )
                ]
            )

        # Phase 3 D-12/D-13/D-14: write run_config.json before any measurements.
        # Position: after _configure_logging (so the INFO line lands in app.log)
        # and after target_countries is set (so provenance reflects the resolved
        # set), but before _bootstrap_vps so a bootstrap crash still leaves a
        # config record on disk.
        self._write_run_config()

        # Draw initial VPs and send for evaluation
        self._bootstrap_vps()

        # Signal-based stop flag (D-05)
        self._stop_event = threading.Event()
        # Wall-clock save tracking (Phase 02.1 D-09) — drives the daily
        # save_daily cadence only. Per-country soft resets are now
        # measurement-count-driven inside RegionalNode.
        self._next_reset_utc: datetime = self._compute_next_midnight()
        # Per-target last-warned timestamps for delayed-result throttling (D-09)
        self._last_delay_warn: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Per-measurement schedule timestamps: country -> {target -> monotonic time}
        self._inflight_times: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Hourly checkpoint timer (D-01 / D-04): all countries checkpoint together
        # via monotonic clock so the cadence is robust to wall-clock jumps.
        self._last_checkpoint_time: float = time.monotonic()
        self._checkpoint_interval: float = 3600.0  # 60 minutes per D-01

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _configure_logging(self) -> None:
        """Install a FileHandler on the root logger pointing at <output_folder>/app.log.

        Must be called before any internal method that emits log records (notably
        _bootstrap_vps). If output_folder is None we install a NullHandler so the
        program does not crash and does not spam stdout (decision: silent-drop
        fallback rather than CWD app.log or stderr).
        """
        root = logging.getLogger()
        # Idempotent: clear any handlers a prior run/import may have attached so
        # repeated Orchestrator instantiations (e.g. tests) don't double-log.
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        root.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        if self.output_folder:
            log_dir = self.output_folder
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "app.log")
            handler = logging.FileHandler(log_path, mode="a")
            handler.setFormatter(formatter)
            root.addHandler(handler)
        else:
            # No output dir => silent. Caller can override later if desired.
            root.addHandler(logging.NullHandler())

    # ------------------------------------------------------------------
    # run-config snapshot (Phase 3 D-12/D-13/D-14)
    # ------------------------------------------------------------------
    def _write_run_config(self) -> None:
        """One-shot run_config.json snapshot at startup (D-12, D-13, D-14).

        Writes <output>/run_config.json with four groups:
          - cli_params:      self.params dict (full CLI parser output)
          - hyperquack:      go_api_endpoint, services, vps_per_country, debug
          - vp_pool_inputs:  ev_file, vp_pool_file, blocklist_file, max_countries,
                             blocked_countries (passed via vp_pool_config kwarg)
          - provenance:      target_countries, action_space_file, git_sha, git_dirty,
                             start_utc_timestamp, python_version

        Atomic write via .tmp + os.replace (matches Phase 02 _write_sidecar
        precedent — T-02-02 / T-03-15 mitigation). Best-effort git provenance:
        failure (non-git env) leaves git_sha=null and git_dirty='unknown'
        without aborting startup. Silently skipped when output_folder is None
        (matches _configure_logging NullHandler fallback).
        """
        if not self.output_folder:
            return

        # Best-effort git provenance (D-13): non-git environments must NOT abort.
        git_sha: Optional[str] = None
        git_dirty: str = "unknown"
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty_out = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            git_dirty = "dirty" if dirty_out else "clean"
        except Exception:
            pass  # non-git env or git not on PATH; leave defaults

        config = {
            "cli_params": self.params,
            "hyperquack": {
                "go_api_endpoint": self._go_api_endpoint,
                "services":        self.services,
                "vps_per_country": self.vps_per_country,
                "debug":           self.api.debug,
            },
            "vp_pool_inputs": self._vp_pool_config or {},
            "provenance": {
                "target_countries":    sorted(self.target_countries),
                "action_space_file":   self.params.get("action_space_file"),
                "git_sha":             git_sha,
                "git_dirty":           git_dirty,
                "start_utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python_version":      sys.version,
            },
        }

        config_path = Path(self.output_folder) / "run_config.json"
        tmp_path = str(config_path) + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            os.replace(tmp_path, str(config_path))
        except Exception as exc:
            logger.error("Failed to write run_config.json: %s", exc)
            return

        short_sha = git_sha[:8] if git_sha else "unknown"
        logger.info(
            "Run config written to %s; git_sha=%s; debug=%s",
            config_path, short_sha, self.api.debug,
        )

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------
    def _bootstrap_vps(self) -> None:
        """Draw initial VPs for each target country and register them for
        evaluation with the Go server."""
        logger.info("Bootstrapping vantage points...")
        for country in self.target_countries:
            drawn = self.vantage_points.get_n_vantages(
                country, self.vps_per_country
            )
            for vp in drawn:
                self.eval_store.register_vp(vp, country)
            if drawn:
                self.api.update_vps(drawn, self.services, tag=country)
                if self.api.debug:
                    self.api._inject_debug_eval_results(drawn)
                logger.info(
                    "Bootstrapped %d VPs for %s: %s",
                    len(drawn), country, drawn,
                )

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def tick(self) -> Dict[str, float]:
        t0 = time.perf_counter()

        # Step 0: move payloads from the server process into local stores
        self.api.receiver.drain_queues()
        t1 = time.perf_counter()

        # Step 1: process VP evaluation results
        self._process_eval_results()
        t2 = time.perf_counter()

        # Per-country sub-step accumulators (Phase 3 D-07: summed across countries).
        agg_time = 0.0
        model_update_time = 0.0
        schedule_time = 0.0

        finished_nodes: List[str] = []

        for country, node in self.agents.items():
            # Step 2: drain raw results, check VP health, aggregate
            _s2 = time.perf_counter()
            raw_results = self.api.drain_raw_results(country)
            if raw_results:
                dropped_vps = self._check_vp_health(country, raw_results)
                if dropped_vps:
                    raw_results = [r for r in raw_results if r.vp not in dropped_vps]
                self._feed_aggregator(country, raw_results)
            agg_time += time.perf_counter() - _s2

            # Step 3: deliver aggregated results to the node
            _s3 = time.perf_counter()
            aggregated = self.api.aggregator.get_ready(country)
            if aggregated:
                node.maybe_update_model(aggregated)
                for r in aggregated:
                    self._inflight_times.get(country, {}).pop(r.target, None)
                    self._last_delay_warn.get(country, {}).pop(r.target, None)
                logger.info(
                    f"{country} received {len(aggregated)} results: {[r.target for r in aggregated]}"
                    )
            model_update_time += time.perf_counter() - _s3

            # Step 4: check for completion (NO timing column per D-08)
            if node.state is NodeState.DONE:
                logger.info(
                    "%s completed %d episodes, finished.",
                    country, node.model.num_episodes,
                )
                finished_nodes.append(country)
                self.completed_agents.add(country)
                continue

            # Step 5: schedule new measurements across all active VPs
            _s5 = time.perf_counter()
            active_vps = self.vantage_points.get_active(country)
            if not active_vps:
                # No active VPs and no inactive replacements — region is dead
                if not self.vantage_points.get_inactive(country):
                    logger.warning(
                        "%s has no remaining vantage points, removing region.",
                        country,
                    )
                    finished_nodes.append(country)
                schedule_time += time.perf_counter() - _s5
                continue

            targets = node.maybe_request_more()
            if not targets:
                schedule_time += time.perf_counter() - _s5
                continue
            logger.info(
                f"{country} requesting {len(targets)} measurements: {targets}"
                )
            # Snapshot the expected VPs at scheduling time so the aggregator
            # knows exactly which VPs should report for each target (D-02).
            # Phase 3 D-04: also pass per-target schedule monotonic time so the
            # aggregator can attach scheduled_at_monotonic to the finalized response.
            now = time.monotonic()
            schedule_times = {t: now for t in targets}
            self.api.aggregator.register_targets(
                country, targets, set(active_vps),
                schedule_times=schedule_times,
            )

            self.api.schedule_measurements(
                vps=active_vps,
                services=self.services,
                targets=targets,
                country=country,
            )
            for t in targets:
                self._inflight_times[country][t] = now
            schedule_time += time.perf_counter() - _s5

        for country in finished_nodes:
            self.agents.pop(country, None)
            self.target_countries.discard(country)

        t_end = time.perf_counter()
        return {
            "drain":           round(t1 - t0, 6),
            "eval_processing": round(t2 - t1, 6),
            "aggregate":       round(agg_time, 6),
            "model_update":    round(model_update_time, 6),
            "schedule":        round(schedule_time, 6),
            "total":           round(t_end - t0, 6),
        }

    def run_forever(self) -> None:
        self._install_signal_handlers()
        iterations = 0

        # Phase 3 D-07/D-09/D-10: per-tick timing CSV writer.
        # File at <output>/tick_timings.csv; opened once at run start, header on
        # first row when file is empty, buffered flush every 10 ticks (~5 s at
        # the 0.5 s sleep cadence), closed before _shutdown.
        timings_path: Optional[Path] = None
        timings_file = None
        timings_writer = None
        timings_fieldnames = [
            "tick_idx", "utc_timestamp", "n_active_countries",
            "drain", "eval_processing", "aggregate",
            "model_update", "schedule", "total",
        ]
        FLUSH_EVERY = 10

        output_folder = getattr(self, "output_folder", None)
        if output_folder:
            try:
                timings_path = Path(output_folder) / "tick_timings.csv"
                # Note: open(..., "a") creates the file if missing, so size==0 on first run.
                timings_file = open(timings_path, "a", newline="")
                timings_writer = csv.DictWriter(timings_file, fieldnames=timings_fieldnames)
                if timings_path.stat().st_size == 0:
                    timings_writer.writeheader()
                    timings_file.flush()
            except Exception as exc:
                logger.error("Failed to open tick_timings.csv: %s", exc)
                timings_writer = None
                if timings_file is not None:
                    try:
                        timings_file.close()
                    except Exception:
                        pass
                timings_file = None

        while not self._stop_event.is_set():
            # Subprocess health check (D-12 / RUNT-06)
            if not self._subprocess_healthy():
                self._shutdown()
                if timings_file is not None:
                    try:
                        timings_file.flush()
                        timings_file.close()
                    except Exception:
                        pass
                return

            # Wall-clock daily save check (Phase 02.1 D-09, D-10)
            # Midnight ONLY fires save_daily — no draining, no soft_reset, no global in-flight clear.
            # Per-country soft resets are now driven by measurement count inside RegionalNode.
            if datetime.now(timezone.utc) >= self._next_reset_utc:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                logger.info(
                    "Midnight UTC reached, saving daily snapshots for %d nodes (date: %s)",
                    len(self.agents), date_str,
                )
                for country, node in self.agents.items():
                    try:
                        state_dir = self._state_dir_for(country)
                        node.save_daily(date_str, state_dir)
                    except Exception as exc:
                        logger.error("Daily save failed for %s: %s", country, exc)
                self._next_reset_utc += timedelta(days=1)

            # Delayed-result logger (D-09 / RUNT-05)
            self._check_delayed_results()

            # Hourly checkpoint (D-01)
            now_mono = time.monotonic()
            if now_mono - self._last_checkpoint_time >= self._checkpoint_interval:
                self._checkpoint_all_nodes()
                self._last_checkpoint_time = now_mono

            # Phase 3 D-09: snapshot tick metadata BEFORE tick() so the row
            # reflects the active-country count at start-of-tick (matches the
            # sub-step accumulators which also iterate over the start-of-tick
            # agents set).
            tick_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            n_countries = len(self.agents)

            step_times = self.tick()
            iterations += 1

            # WR-07 fix: if the writer is dead (e.g. transient I/O error
            # during initial header write or a previous writerow), retry
            # reopening on each tick rather than silently disabling tick
            # timings for the rest of the run. This restores OBSV-01's
            # "files are created on first row" invariant after recoverable
            # failures (network-mounted output folder hiccups, etc.).
            if timings_writer is None and output_folder:
                try:
                    timings_path = Path(output_folder) / "tick_timings.csv"
                    timings_file = open(timings_path, "a", newline="")
                    timings_writer = csv.DictWriter(timings_file, fieldnames=timings_fieldnames)
                    if timings_path.stat().st_size == 0:
                        timings_writer.writeheader()
                        timings_file.flush()
                    logger.info(
                        "tick_timings.csv reopened successfully at tick %d", iterations,
                    )
                except Exception as exc:
                    # Recurring warning every FLUSH_EVERY ticks while the
                    # writer remains dead, so the operator sees the issue.
                    if iterations % FLUSH_EVERY == 1:
                        logger.error(
                            "tick_timings.csv reopen attempt failed at tick %d: %s",
                            iterations, exc,
                        )
                    timings_writer = None
                    if timings_file is not None:
                        try:
                            timings_file.close()
                        except Exception:
                            pass
                    timings_file = None

            if timings_writer is not None:
                row = {
                    "tick_idx":           iterations,
                    "utc_timestamp":      tick_utc,
                    "n_active_countries": n_countries,
                    **step_times,
                }
                try:
                    timings_writer.writerow(row)
                    if iterations % FLUSH_EVERY == 0:
                        timings_file.flush()
                except Exception as exc:
                    logger.error("tick_timings write failed at tick %d: %s", iterations, exc)
                    # WR-07 fix: if the write itself fails (not just the
                    # initial open), drop the writer so the reopen-retry
                    # path above gets a chance on the next tick. Do not
                    # leave a half-broken handle in place for the rest
                    # of the run.
                    try:
                        timings_file.close()
                    except Exception:
                        pass
                    timings_writer = None
                    timings_file = None

            sleep(0.5)

        # Graceful exit path: flush + close before _shutdown.
        if timings_file is not None:
            try:
                timings_file.flush()
                timings_file.close()
            except Exception as exc:
                logger.error("Failed to close tick_timings.csv: %s", exc)

        self._shutdown()

    # ------------------------------------------------------------------
    # signal handling (D-04 / D-11)
    # ------------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT handlers. Must be called from main thread."""
        def _handler(signum, frame):
            logger.info("Signal %d received, initiating shutdown", signum)
            self._stop_event.set()

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _shutdown(self) -> None:
        """Write stats for all active nodes and log summary (D-11).

        Also saves a rolling checkpoint per node (D-03 / D-04) so that graceful
        shutdown loses zero learning. Each save is wrapped in its own try/except
        so one failing node does not block the others.
        """
        logger.info("Shutting down — writing stats for %d active nodes", len(self.agents))
        for country, node in self.agents.items():
            try:
                node.write_stats()
            except Exception as exc:
                logger.error("Failed to write stats for %s: %s", country, exc)
            try:
                state_dir = self._state_dir_for(country)
                node.save_checkpoint(state_dir)
            except Exception as exc:
                logger.error("Failed to checkpoint %s on shutdown: %s", country, exc)
            try:
                node.close_measurements()
            except Exception as exc:
                logger.error(
                    "Failed to close measurements writer for %s: %s",
                    country, exc,
                )
        logger.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # subprocess health (D-12 / RUNT-06)
    # ------------------------------------------------------------------
    def _subprocess_healthy(self) -> bool:
        """Check if the FastAPI receiver subprocess is still alive."""
        proc = self.api.receiver._process
        if proc is None:
            return True  # debug mode — no subprocess
        if not proc.is_alive():
            logger.error(
                "FastAPI receiver subprocess exited unexpectedly (exitcode=%s), shutting down",
                proc.exitcode,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # wall-clock save (Phase 02.1 D-09)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_next_midnight() -> datetime:
        """Compute the next midnight UTC boundary."""
        now = datetime.now(timezone.utc)
        today_midnight = datetime.combine(now.date(), dtime(0, 0, 0), tzinfo=timezone.utc)
        return today_midnight + timedelta(days=1)

    # ------------------------------------------------------------------
    # state persistence (D-01, D-02, D-04 + Phase 02.1 D-12)
    # ------------------------------------------------------------------
    def _state_dir_for(self, country: str) -> Path:
        """Return the state subdirectory for a country's checkpoint files.

        Mirrors the per-country path construction used in RegionalNode.__init__:
        spaces in country names become underscores so the path is filesystem-safe.
        """
        return Path(self.output_folder) / country.replace(" ", "_") / "state"

    def _checkpoint_all_nodes(self) -> None:
        """Save rolling checkpoint for all active nodes (per D-01, D-02).

        Each node is wrapped in its own try/except so a single failing node never
        prevents the others from checkpointing (T-02-06 mitigation).
        """
        logger.info("Checkpointing %d active nodes", len(self.agents))
        for country, node in self.agents.items():
            try:
                state_dir = self._state_dir_for(country)
                node.save_checkpoint(state_dir)
            except Exception as exc:
                logger.error("Checkpoint failed for %s: %s", country, exc)

    def _on_country_reset(self, country: str) -> None:
        """Per-country cleanup callback fired by RegionalNode after a per-iteration
        soft reset (Phase 02.1 D-12).

        Clears ONLY this country's in-flight tracking entries — other countries'
        entries are untouched so they continue to drain, schedule, and progress
        their own iterations independently. This is the per-country counterpart
        to the global ``_inflight_times.clear()`` / ``_last_delay_warn.clear()``
        that the legacy Phase 02 midnight-reset path used to perform (which
        incorrectly cleared every country's tracking on each midnight under the
        old semantics).

        WR-05/WR-09 partial mitigation: explicitly log how many in-flight
        targets are being discarded at the iteration boundary so the operator
        can see the metric drift. Note that ``_finish_episode_if_ready`` only
        triggers when ``in_flight == 0``, so any non-zero count here indicates
        the schedule path incremented in_flight but the aggregated result was
        already drained before this callback fired — or, more concerning, that
        FastAPI-subprocess results that arrived after the boundary will land
        on an empty tracking dict and silently no-op the .pop() in tick(). The
        proper fix per WR-05 is iteration-tagged in-flight maps; this log
        ensures the issue is at least observable.
        """
        discarded_inflight = len(self._inflight_times.get(country, {}))
        discarded_warns = len(self._last_delay_warn.get(country, {}))
        self._inflight_times.pop(country, None)
        self._last_delay_warn.pop(country, None)
        if discarded_inflight or discarded_warns:
            logger.info(
                "%s: iteration-boundary discarded %d in-flight tracking entries, "
                "%d delay-warn entries (results arriving after this point will "
                "no-op the tick() pop and the delayed-result metric will under-count)",
                country, discarded_inflight, discarded_warns,
            )
        else:
            logger.debug("%s: cleared in-flight tracking after iteration reset", country)

    # ------------------------------------------------------------------
    # delayed-result logger (D-09 / RUNT-05)
    # ------------------------------------------------------------------
    def _check_delayed_results(self) -> None:
        """Log measurements that have been in-flight longer than 5 minutes.
        First warning at 5min, then re-warn every 5min with updated wait time.
        When stuck targets are found, trigger /debug poll to detect and resend
        lost measurements (D-03)."""
        now = time.monotonic()
        threshold = 300  # 5 minutes
        stuck_by_country: Dict[str, List[str]] = defaultdict(list)

        for country, targets in self._inflight_times.items():
            for target, scheduled in targets.items():
                elapsed = now - scheduled
                if elapsed < threshold:
                    continue
                last_warned = self._last_delay_warn.get(country, {}).get(target)
                if last_warned is not None and (now - last_warned) < threshold:
                    continue
                elapsed_min = elapsed / 60.0
                logger.warning(
                    "DELAYED: %s target %s waiting %.1fmin",
                    country, target, elapsed_min,
                )
                self._last_delay_warn[country][target] = now
                stuck_by_country[country].append(target)

        if stuck_by_country:
            self._check_and_resend_stuck(stuck_by_country)

    def _check_and_resend_stuck(self, stuck_by_country: Dict[str, List[str]]) -> None:
        """One /debug call per check cycle; resend measurements lost by Hyperquack (D-03).

        For each stuck target, check if the VP the aggregator is waiting on
        still has the target in its unstarted_work queue. If not, the measurement
        was lost -- resend to the same VP, or to an alternative if the VP is
        pending_removal (D-05).
        """
        if self.api.debug:
            return

        debug_response = self.api.call_debug_endpoint()
        if not isinstance(debug_response, list):
            logger.warning("Unexpected /debug response format: %s", type(debug_response))
            return

        # Build lookup: ip -> VP debug info
        vp_status = {entry["ip"]: entry for entry in debug_response}

        for country, stuck_targets in stuck_by_country.items():
            active_vps = self.vantage_points.get_active(country)

            # WR-03 fix: snapshot aggregator state via the public API instead
            # of reaching into private attributes. The aggregator owns its own
            # locking; we receive immutable frozenset values that remain safe
            # to use after the call returns.
            pending_snapshot = self.api.aggregator.snapshot_pending(
                country, stuck_targets,
            )

            resend_list = []
            for target in stuck_targets:
                expected, received = pending_snapshot.get(
                    target, (frozenset(), frozenset()),
                )
                missing_vps = expected - received

                for vp_ip in missing_vps:
                    info = vp_status.get(vp_ip)
                    if info is None:
                        logger.warning("VP %s not in /debug response, assuming lost", vp_ip)
                        resend_list.append((target, vp_ip, False))
                        continue

                    # Check if VP still has this target in its unstarted_work
                    unstarted_keywords = {w["keyword"] for w in info.get("unstarted_work", [])}
                    if target in unstarted_keywords:
                        continue  # VP still has it queued, not lost yet

                    is_pending_removal = info.get("pending_removal", False)
                    resend_list.append((target, vp_ip, is_pending_removal))

            # Now resend outside the lock
            for target, vp_ip, is_pending_removal in resend_list:
                send_to = vp_ip
                if is_pending_removal:
                    alternatives = [v for v in active_vps if v != vp_ip]
                    if not alternatives:
                        logger.warning("No alternative VP for lost %s/%s", country, target)
                        continue
                    send_to = alternatives[0]

                logger.warning("Resending lost measurement %s/%s to %s (original VP: %s)",
                               country, target, send_to, vp_ip)
                self.api.schedule_measurements(
                    vps=[send_to],
                    services=self.services,
                    targets=[target],
                    country=country,
                )

    # ------------------------------------------------------------------
    # VP evaluation
    # ------------------------------------------------------------------
    def _process_eval_results(self) -> None:
        """Drain the eval store and handle each VP evaluation."""
        eval_results = self.eval_store.drain()
        failed_vps: List[str] = []
        for payload in eval_results:
            vp = payload.vp
            country = self.eval_store.get_country(vp)
            if country is None:
                logger.warning("Received eval for unknown VP %s, ignoring", vp)
                continue

            if payload.issue is None and payload.template is not None:
                # VP is good — keep it active
                self.vantage_points.confirm_active(country, vp)
                logger.info("VP %s confirmed OK for %s", vp, country)

                # Create node lazily on first good VP
                if country not in self.agents and country not in self.completed_agents:
                    self.agents[country] = RegionalNode(
                        params=self.params,
                        country_name=country,
                        model_klass=BatchUCB,
                        output_folder=self.output_folder,
                        action_space_folder=self.previous_values_folder,
                        batch_size=5,
                        on_reset=self._on_country_reset,
                    )
                # NOTE: aggregator expected VPs are updated at scheduling
                # time, not here, to avoid snapshot mismatches with
                # in-flight targets.
            else:
                # VP failed — reject and try replacement
                replacement = self.vantage_points.reject_vp(country, vp)
                failed_vps.append(vp)
                if replacement:
                    self.eval_store.register_vp(replacement, country)
                    self.api.update_vps([replacement], self.services, tag=country)
                    if self.api.debug:
                        self.api._inject_debug_eval_results([replacement])
                    # logger.info(
                    #     "VP %s failed for %s, replacement %s sent for eval",
                    #     vp, country, replacement,
                    # )
                else:
                    logger.warning(
                        "VP %s failed for %s, no replacements available",
                        vp, country,
                    )
                    # If the country has no VPs left at all, abandon it
                    if (not self.vantage_points.get_active(country)
                            and not self.vantage_points.get_inactive(country)):
                        logger.warning(
                            "%s has no remaining vantage points, removing region.",
                            country,
                        )
                        self.target_countries.discard(country)

                # Drop the failed VP from pending aggregator entries so
                # in-flight targets aren't stuck waiting for it.
                self.api.aggregator.drop_vp(country, vp)

        # Batch-remove all failed VPs from Hyperquack (D-09)
        if failed_vps:
            self.api.remove_vantage_points(failed_vps, expect_unstarted=True)

    # ------------------------------------------------------------------
    # measurement result processing
    # ------------------------------------------------------------------
    def _feed_aggregator(
        self, country: str, raw_results: List[TestPayload]
    ) -> None:
        """Parse raw TestPayloads and feed into the aggregator."""
        for r in raw_results:
            blocked = False
            if r.response:
                blocked = not r.response[0].matches_template
            blocked = r.stateful_block or blocked
            self.api.aggregator.record(country, r.vp, r.test_url, blocked)

    def _check_vp_health(
        self, country: str, raw_results: List[TestPayload]
    ) -> Set[str]:
        """Track VP control failures and remove unhealthy VPs.

        Returns the set of VP IPs that were dropped during this call.
        """
        dropped_vps: Set[str] = set()
        for r in raw_results:
            key = (country, r.vp)
            if r.controls_failed:
                self._vp_failure_counts[key] += 1
                if self._vp_failure_counts[key] >= VP_FAILURE_THRESHOLD:
                    logger.warning(
                        "VP %s in %s exceeded failure threshold, removing",
                        r.vp, country,
                    )
                    dropped_vps.add(r.vp)
                    replacement = self.vantage_points.reject_vp(country, r.vp)
                    self.api.aggregator.drop_vp(country, r.vp)
                    self.api.remove_vantage_points([r.vp], expect_unstarted=True)
                    if replacement:
                        self.eval_store.register_vp(replacement, country)
                        self.api.update_vps([replacement], self.services, tag=country)
                        if self.api.debug:
                            self.api._inject_debug_eval_results([replacement])
                    del self._vp_failure_counts[key]
            else:
                # Reset counter on success
                self._vp_failure_counts.pop(key, None)
        return dropped_vps

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _has_pending_evals(self) -> bool:
        """Return True if any target country still lacks a RegionalNode
        and might still receive VP evaluation results."""
        return any(c not in self.agents for c in self.target_countries)


class OrchestrationParser(UCBNaiveParserOptions):

    def set_params(self, args):
        if args.outfile[-1] != "/":
            args.outfile += "/"
        super().set_params(args)


if __name__ == "__main__":
    parser = OrchestrationParser()
    params = parser.parse()

    # Phase 3 D-13: vp_pool_inputs group of run_config.json. Built from the
    # same literals consumed by VantagePoints below so the snapshot is the
    # single source of truth for run reproducibility.
    vp_pool_config = {
        "ev_file":           "local/ev-certs.csv",
        "vp_pool_file":      "local/vp_pool.csv",
        "blocklist_file":    "local/blocklist.txt",
        "max_countries":     15,
        "blocked_countries": [],
    }
    vp_pool = VantagePoints(**vp_pool_config)

    m = Orchestrator(
        params=params,
        vantage_points=vp_pool,
        go_api_endpoint="http://127.0.0.1:8888",
        services=["https"],
        vps_per_country=3,
        vp_pool_config=vp_pool_config,
    )
    m.run_forever()
    # python3 Infrastructure/main/orchestrator.py -E 1 -m 1000 -v -f "categories" -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv -f "categories" -s 0.0 -c 0.03 -V 0.0 -o outputs/outtest
