import requests
import json
from Infrastructure.apis.api import Api
from dataclasses import dataclass, asdict
import subprocess
from typing import Dict, List, Set
import sys
from queue import Queue
import threading
import uvicorn
import time
import random
import logging
from datetime import datetime, timezone
from Infrastructure.utils.store import MeasurementStore
from Infrastructure.utils.eval_store import EvalStore
from Infrastructure.utils.aggregator import MeasurementAggregator
from Infrastructure.apis.server import MeasurementReceiver
from Infrastructure.utils.structures import (
    TestPayload,
    EvalPayload,
    TestResponseData,
    LocationData,
    Tag,
    Job,
    MeasurementResponse,
)

logger = logging.getLogger(__name__)

class HyperQuackAPI(Api):
    """
    Lightweight API interface for a reinforcement learning model that
    can communicate with the Go-based service API.
    """

    def __init__(
        self,
        go_api_url: str,
        eval_store: EvalStore = None,
        vantage_points=None,
        debug: bool = False,
        debug_block_prob: float = 0.15,
        debug_min_delay_s: float = 0.0,
        debug_max_delay_s: float = 0.0,
    ):
        """
        :param go_api_url: Base URL of the Go API (e.g. http://127.0.0.1:8080)
        :param eval_store: Shared EvalStore for VP evaluation results
        :param vantage_points: VantagePoints instance for port lookups
        """
        self.go_api_url = go_api_url.rstrip("/")
        self.retries = 5
        self.vps = set()
        self.tags = set()
        self.vantage_points = vantage_points
        self.store: MeasurementStore = MeasurementStore()
        self.eval_store = eval_store
        self.aggregator = MeasurementAggregator()
        self.receiver: MeasurementReceiver = MeasurementReceiver(
            self.store, eval_store=self.eval_store
        )
        self.debug = debug
        if not self.debug:
            self.receiver.start_in_background()
        else:
            self.debug_block_prob = debug_block_prob
            self.debug_min_delay_s = debug_min_delay_s
            self.debug_max_delay_s = debug_max_delay_s
            logger.info("Starting measurement API in DEBUG mode...")

    def update_aggregator_vps(self, country: str, vps: Set[str]) -> None:
        """Inform the aggregator which VPs are active for a country."""
        self.aggregator.set_expected_vps(country, vps)

    def schedule_measurements(
        self, vps: List[str], services: List[str], targets: List[str], country: str
    ):
        update_response = self.update_vps(vps, services)
        if (
            "invalid_entries" in update_response
            and len(update_response["invalid_entries"]) != 0
        ):
            print(
                f"Invalid entries: {update_response['invalid_entries']}",
                file=sys.stderr,
            )

        # Create Jobs with per-VP services (port-aware)
        jobs = []
        for vp in vps:
            vp_services = (
                self.vantage_points.get_services(vp, services)
                if self.vantage_points
                else services
            )
            for target in targets:
                jobs.append(Job(target, vp_services, vp, ""))

        if self.debug:
            self._inject_debug_results(country=country, jobs=jobs)
            return

        self.add_work(jobs)
        return

    def try_get_results(self, country: str) -> List[MeasurementResponse]:
        results = self.store.get_country_batch(country=country)
        return self.parse_measurements(results)

    def drain_raw_results(self, country: str) -> List[TestPayload]:
        """Return raw TestPayloads for *country* without parsing.

        The orchestrator uses this to inspect per-VP results (e.g. for
        health monitoring) before feeding them into the aggregator.
        """
        return self.store.get_country_batch(country=country)

    # ---------------------------- Helpers -----------------------------
    def parse_measurements(
        self, results: List[TestPayload]
    ) -> List[MeasurementResponse]:
        parsed_output = []
        for r in results:
            target = r.test_url
            blocked = False
            if r.response:
                blocked = not r.response[0].matches_template
            parsed_output.append(
                MeasurementResponse(
                    target=target, blocked=(r.stateful_block or blocked)
                )
            )
        return parsed_output

    def update_vps(self, new_vps: List[str], services: List[str]):
        new_vps = [vp for vp in new_vps if vp not in self.vps]
        if len(new_vps) == 0:
            return {}
        for vp in new_vps:
            self.vps.add(vp)
        if self.debug:
            return {}
        return self.add_vantage_points(new_vps, services)

    # ---------------------------- CALLS -----------------------------
    def add_vantage_points(self, ips: List[str], services: List[str]):
        endpoint = "/add-vantage-points"
        vp_entries = []
        for ip in ips:
            vp_services = (
                self.vantage_points.get_services(ip, services)
                if self.vantage_points
                else services
            )
            entry = {"ip": ip, "services": vp_services}
            vp_entries.append(entry)
        body = {"vantage_points": vp_entries}
        # logging.info(f"Adding vantage points with body\n{body}\n")
        for ip in ips:
            if ip not in self.vps:
                self.vps.add(ip)
        response = self.call_go_api(endpoint, body)
        # logging.info(f"Received response\n{response}")
        return response

    def add_tags(self, tags: List[Tag]):
        endpoint = "/add-tags"
        body = {"tags": [asdict(t) for t in tags]}
        return self.call_go_api(endpoint, body)

    def add_work(self, jobs: List[Job]):
        if not self.debug:
            endpoint = "/add-work"
            body = {"work": [asdict(j) for j in jobs]}
            return self.call_go_api(endpoint, body)
        return

    def _inject_debug_results(self, country: str, jobs: List["Job"]) -> None:
        """
        Simulate measurement completion by pushing synthetic TestPayloads
        into the store.
        """

        def worker():
            # optional simulated delay
            if self.debug_max_delay_s > 0:
                time.sleep(
                    random.uniform(self.debug_min_delay_s, self.debug_max_delay_s)
                )

            now = datetime.now(timezone.utc).isoformat()

            for j in jobs:
                blocked = random.random() < self.debug_block_prob

                payload = TestPayload(
                    vp=j.vantage_point_predicate,
                    location=LocationData(country_name=country, country_code="XX"),
                    service=(j.services[0] if j.services else "https"),
                    test_url=j.keyword,
                    response=[
                        TestResponseData(
                            matches_template=(not blocked),
                            start_time=now,
                            end_time=now,
                        )
                    ],
                    anomaly=False,
                    controls_failed=False,
                    stateful_block=blocked,
                )
                self.store.record_result(country, payload)

        threading.Thread(target=worker, daemon=True).start()

    def _inject_debug_eval_results(
        self, vps: List[str], success_prob: float = 0.9
    ) -> None:
        """Simulate VP evaluation for debug mode."""

        def worker():
            for vp in vps:
                ok = random.random() < success_prob
                payload = EvalPayload(
                    vp=vp,
                    service="https",
                    response=[],
                    issue=None if ok else "debug_failure",
                    template={"debug": True} if ok else None,
                )
                if self.eval_store:
                    self.eval_store.record(payload)

        threading.Thread(target=worker, daemon=True).start()

    def call_debug_endpoint(self):
        endpoint = "/debug"
        return self.call_go_api(endpoint, method="GET")

    # ---------------------------- HELPERS ----------------------------
    def call_go_api(self, endpoint: str, data: dict = {}, method: str = "POST"):
        """Send a POST request to the Go API and return JSON response."""
        method = method.upper()
        if method not in ["POST", "GET"]:
            logging.warning(f"Invalid method: {method}")
            return {"error": "invalid method"}
        retries = 0
        for _ in range(self.retries):
            try:
                url = f"{self.go_api_url}{endpoint}"
                response = requests.post(url, json=data, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                retries += 1
                if retries == self.retries:
                    logging.warning(f"[Error] Failed to call Go API: {e}")
                    return {"error": str(e)}
                else:
                    continue
