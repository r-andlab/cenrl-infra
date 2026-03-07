import logging
import os
from collections import defaultdict
from pathlib import Path
from time import sleep
from typing import Dict, List, Tuple

from models.ucb.ucb_naive import UCBNaiveParserOptions
from Infrastructure.apis.funneler import HyperQuackAPI
from Infrastructure.models.BatchUCB import BatchUCB
from Infrastructure.main.node import RegionalNode
from Infrastructure.main.vantage_points import VantagePoints
from Infrastructure.utils.eval_store import EvalStore
from Infrastructure.utils.structures import NodeState, TestPayload

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
            go_api_endpoint, eval_store=self.eval_store, debug=debug
        )

        # Nodes are created lazily once a VP passes evaluation.
        self.agents: Dict[str, RegionalNode] = {}

        # VP health tracking: (country, vp) -> consecutive failure count
        self._vp_failure_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        # Determine target countries
        if countries is None:
            countries = self.vantage_points.countries()
        self.target_countries = countries

        # Draw initial VPs and send for evaluation
        self._bootstrap_vps()

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
                self.api.update_vps(drawn, self.services)
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
        # Step 1: process VP evaluation results
        self._process_eval_results()

        finished_nodes: List[str] = []

        for country, node in self.agents.items():
            # Step 2: drain raw results, check VP health, aggregate
            raw_results = self.api.drain_raw_results(country)
            if raw_results:
                self._check_vp_health(country, raw_results)
                self._feed_aggregator(country, raw_results)

            # Step 3: deliver aggregated results to the node
            aggregated = self.api.aggregator.get_ready(country)
            if aggregated:
                node.maybe_update_model(aggregated)

            # Step 4: check for completion
            if node.state is NodeState.DONE:
                logger.info(
                    "%s completed %d episodes, finished.",
                    country, node.model.num_episodes,
                )
                finished_nodes.append(country)
                continue

            # Step 5: schedule new measurements across all active VPs
            active_vps = self.vantage_points.get_active(country)
            if not active_vps:
                continue

            targets = node.maybe_request_more()
            if not targets:
                continue

            self.api.schedule_measurements(
                vps=active_vps,
                services=self.services,
                targets=targets,
                country=country,
            )

        for country in finished_nodes:
            self.agents.pop(country, None)

    def run_forever(self) -> None:
        while self.agents or self._has_pending_evals():
            self.tick()
            sleep(0.5)
        logger.info("All countries completed, exiting program...")

    # ------------------------------------------------------------------
    # VP evaluation
    # ------------------------------------------------------------------
    def _process_eval_results(self) -> None:
        """Drain the eval store and handle each VP evaluation."""
        eval_results = self.eval_store.drain()
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
                if country not in self.agents:
                    self.agents[country] = RegionalNode(
                        params=self.params,
                        country_name=country,
                        vps=self.vantage_points.get_active(country),
                        model_klass=BatchUCB,
                        output_folder=self.output_folder,
                        action_space_folder=self.previous_values_folder,
                    )

                # Sync aggregator's expected VP set
                self.api.update_aggregator_vps(
                    country,
                    set(self.vantage_points.get_active(country)),
                )
            else:
                # VP failed — reject and try replacement
                replacement = self.vantage_points.reject_vp(country, vp)
                if replacement:
                    self.eval_store.register_vp(replacement, country)
                    self.api.update_vps([replacement], self.services)
                    if self.api.debug:
                        self.api._inject_debug_eval_results([replacement])
                    logger.info(
                        "VP %s failed for %s, replacement %s sent for eval",
                        vp, country, replacement,
                    )
                else:
                    logger.warning(
                        "VP %s failed for %s, no replacements available",
                        vp, country,
                    )

                # Update aggregator in case VP was already expected
                active = self.vantage_points.get_active(country)
                if active:
                    self.api.update_aggregator_vps(country, set(active))
                    self.api.aggregator.drop_vp(country, vp)

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
    ) -> None:
        """Track VP control failures and remove unhealthy VPs."""
        for r in raw_results:
            key = (country, r.vp)
            if r.controls_failed:
                self._vp_failure_counts[key] += 1
                if self._vp_failure_counts[key] >= VP_FAILURE_THRESHOLD:
                    logger.warning(
                        "VP %s in %s exceeded failure threshold, removing",
                        r.vp, country,
                    )
                    replacement = self.vantage_points.reject_vp(country, r.vp)
                    self.api.aggregator.drop_vp(country, r.vp)
                    active = self.vantage_points.get_active(country)
                    self.api.update_aggregator_vps(country, set(active))
                    if replacement:
                        self.eval_store.register_vp(replacement, country)
                        self.api.update_vps([replacement], self.services)
                        if self.api.debug:
                            self.api._inject_debug_eval_results([replacement])
                    del self._vp_failure_counts[key]
            else:
                # Reset counter on success
                self._vp_failure_counts.pop(key, None)

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
        ev_file="ev-certs.csv",
        blocklist_file="blocklist.txt",
        max_size=10,
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
