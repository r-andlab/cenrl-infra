import logging
import os
import signal
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta, time as dtime
from pathlib import Path
from time import sleep
from typing import Dict, List, Set, Tuple

from models.ucb.ucb_naive import UCBNaiveParserOptions
from Infrastructure.apis.funneler import HyperQuackAPI
from Infrastructure.models.BatchUCB import BatchUCB
from Infrastructure.main.node import RegionalNode
from Infrastructure.main.vantage_points import VantagePoints
from Infrastructure.utils.eval_store import EvalStore
from Infrastructure.utils.structures import NodeState, TestPayload, Tag

logging.basicConfig(level=logging.INFO)
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
    ):
        self.params = params
        self.output_folder = params.get("output_directory", None)
        self.previous_values_folder = previous_values_folder
        if previous_values_folder:
            path = Path(previous_values_folder)
            if not path.exists() or not path.is_dir():
                self.previous_values_folder = None
        if self.output_folder:
            os.makedirs(os.path.dirname(self.output_folder), exist_ok=True)

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
                        result_output_file=f"{country}_result.json",
                        eval_output_file=f"{country}_eval.json",
                    )
                ]
            )

        # Draw initial VPs and send for evaluation
        self._bootstrap_vps()

        # Signal-based stop flag (D-05)
        self._stop_event = threading.Event()
        # Wall-clock reset tracking (D-01)
        self._next_reset_utc: datetime = self._compute_next_midnight()
        self._draining: bool = False
        # Per-target last-warned timestamps for delayed-result throttling (D-09)
        self._last_delay_warn: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Per-measurement schedule timestamps: country -> {target -> monotonic time}
        self._inflight_times: Dict[str, Dict[str, float]] = defaultdict(dict)

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
    def tick(self) -> None:
        # Step 0: move payloads from the server process into local stores
        self.api.receiver.drain_queues()

        # Step 1: process VP evaluation results
        self._process_eval_results()

        finished_nodes: List[str] = []

        for country, node in self.agents.items():
            # Step 2: drain raw results, check VP health, aggregate
            raw_results = self.api.drain_raw_results(country)
            if raw_results:
                dropped_vps = self._check_vp_health(country, raw_results)
                if dropped_vps:
                    raw_results = [r for r in raw_results if r.vp not in dropped_vps]
                self._feed_aggregator(country, raw_results)

            # Step 3: deliver aggregated results to the node
            aggregated = self.api.aggregator.get_ready(country)
            if aggregated:
                node.maybe_update_model(aggregated)
                for r in aggregated:
                    self._inflight_times.get(country, {}).pop(r.target, None)
                    self._last_delay_warn.get(country, {}).pop(r.target, None)
                logger.info(
                    f"{country} received {len(aggregated)} results: {[r.target for r in aggregated]}"
                    )

            # Step 4: check for completion
            if node.state is NodeState.DONE:
                logger.info(
                    "%s completed %d episodes, finished.",
                    country, node.model.num_episodes,
                )
                finished_nodes.append(country)
                self.completed_agents.add(country)
                continue

            # Step 5: schedule new measurements across all active VPs
            # Guard: skip scheduling while draining for midnight reset (D-02)
            if self._draining:
                continue

            active_vps = self.vantage_points.get_active(country)
            if not active_vps:
                # No active VPs and no inactive replacements — region is dead
                if not self.vantage_points.get_inactive(country):
                    logger.warning(
                        "%s has no remaining vantage points, removing region.",
                        country,
                    )
                    finished_nodes.append(country)
                continue

            targets = node.maybe_request_more()
            if not targets:
                continue
            logger.info(
                f"{country} requesting {len(targets)} measurements: {targets}"
                )
            # Snapshot the expected VPs at scheduling time so the aggregator
            # knows exactly which VPs should report for each target (D-02).
            self.api.aggregator.register_targets(country, targets, set(active_vps))

            self.api.schedule_measurements(
                vps=active_vps,
                services=self.services,
                targets=targets,
                country=country,
            )
            now = time.monotonic()
            for t in targets:
                self._inflight_times[country][t] = now

        for country in finished_nodes:
            if self._draining:
                continue  # defer removal until after soft reset (Pitfall 5)
            self.agents.pop(country, None)
            self.target_countries.discard(country)

    def run_forever(self) -> None:
        self._install_signal_handlers()
        sum_time = []
        iterations = 0
        while not self._stop_event.is_set():
            # Subprocess health check (D-12 / RUNT-06)
            if not self._subprocess_healthy():
                self._shutdown()
                return

            # Wall-clock daily reset check (D-01 / D-02)
            if datetime.now(timezone.utc) >= self._next_reset_utc:
                if not self._draining:
                    logger.info("Midnight UTC reached, entering drain-before-reset mode")
                    self._draining = True
                if self._all_nodes_drained():
                    self._perform_soft_reset()
                    self._next_reset_utc += timedelta(days=1)
                    self._draining = False

            # Delayed-result logger (D-09 / RUNT-05)
            self._check_delayed_results()

            start_time = time.perf_counter()
            self.tick()
            end_time = time.perf_counter()
            sum_time.append(end_time - start_time)
            iterations += 1
            sleep(0.5)

        self._shutdown()
        if iterations > 0:
            logger.info(
                "Average time for tick function to complete: %.6f",
                sum(sum_time) / iterations,
            )

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
        """Write stats for all active nodes and log summary (D-11)."""
        logger.info("Shutting down — writing stats for %d active nodes", len(self.agents))
        for country, node in self.agents.items():
            try:
                node.write_stats()
            except Exception as exc:
                logger.error("Failed to write stats for %s: %s", country, exc)
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
    # wall-clock reset (D-01 / D-02 / D-03)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_next_midnight() -> datetime:
        """Compute the next midnight UTC boundary."""
        now = datetime.now(timezone.utc)
        today_midnight = datetime.combine(now.date(), dtime(0, 0, 0), tzinfo=timezone.utc)
        return today_midnight + timedelta(days=1)

    def _all_nodes_drained(self) -> bool:
        """Return True if all nodes have zero in-flight measurements."""
        return all(node.in_flight == 0 for node in self.agents.values())

    def _perform_soft_reset(self) -> None:
        """Call soft_reset() on every active node (D-03)."""
        logger.info("Performing daily soft reset for %d nodes", len(self.agents))
        for country, node in self.agents.items():
            node.soft_reset()
        self._inflight_times.clear()
        self._last_delay_warn.clear()

    # ------------------------------------------------------------------
    # delayed-result logger (D-09 / RUNT-05)
    # ------------------------------------------------------------------
    def _check_delayed_results(self) -> None:
        """Log measurements that have been in-flight longer than 5 minutes.
        First warning at 5min, then re-warn every 5min with updated wait time."""
        now = time.monotonic()
        threshold = 300  # 5 minutes
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

            if payload.template is not None:
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
                        batch_size=5
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
                    self.api.update_vps([replacement], self.services)
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
            self.api.remove_vantage_points(failed_vps)

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
                    self.api.remove_vantage_points([r.vp])
                    if replacement:
                        self.eval_store.register_vp(replacement, country)
                        self.api.update_vps([replacement], self.services)
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
    vp_pool = VantagePoints(
        ev_file="local/ev-certs.csv",
        vp_pool_file="local/vp_pool.csv",
        blocklist_file="local/blocklist.txt",
        max_countries=5,
        blocked_countries=["India"],
    )
    m = Orchestrator(
        params=params,
        vantage_points=vp_pool,
        go_api_endpoint="http://127.0.0.1:8888",
        services=["https"],
        vps_per_country=3,
    )
    m.run_forever()
    # python3 Infrastructure/main/orchestrator.py -E 1 -m 1000 -v -f "categories" -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv -f "categories" -s 0.0 -c 0.03 -V 0.0 -o outputs/outtest
