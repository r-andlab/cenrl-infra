import requests
import json
from Infrastructure.apis.api import Api
from dataclasses import dataclass, asdict
import subprocess
from typing import Any, Dict, List, Tuple, Optional
import sys
from queue import Queue
import threading
import uvicorn
import time
import random
from datetime import datetime, timezone
from Infrastructure.utils.store import MeasurementStore
from Infrastructure.apis.server import MeasurementReceiver
from Infrastructure.utils.structures import (
    RequestPayload,
    ResponseData,
    LocationData,
    Tag,
    Job,
    MeasurementResponse,
)


class SmallQuackAPI(Api):
    def __init__(
        self, go_file: str = "/Users/grahamklingler/Repositories/hyperquackv2/main.go"
    ):
        self.go_file = go_file

    def test(self, target):
        """Returns true if target blocked"""
        measurement = self.run_go_trial(keyword=target, port=443)
        return not measurement["response"][0]["matches_template"]

    def run_go_trial(
        self,
        keyword: str,
        ip: str = "12.47.31.201",
        port: int = 443,
    ) -> Dict[str, Any]:
        """
        Run the Go trial program and return its JSON result as a Python dict.

        :param keyword: keyword for trial (-keyword flag)
        :param ip: target IP address (-ip flag)
        :param port: target port (-port flag)
        :param go_main_path: path to the Go main file (for `go run`)
                            or path to a compiled binary if you change the command.
        """
        cmd = [
            "go",
            "run",
            self.go_file,
            "-keyword",
            keyword,
            "-ip",
            ip,
            "-port",
            str(port),
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Go trial failed with exit code {proc.returncode}:\n{proc.stderr}"
            )

        # Grab the last non-empty line (in case Go prints logs + JSON)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError("Go trial produced no output")

        raw = lines[-1]

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to decode Go output as JSON:\n{raw}\nError: {e}"
            ) from e


class HyperQuackAPI(Api):
    """
    Lightweight API interface for a reinforcement learning model that
    can communicate with the Go-based service API.
    """

    def __init__(
        self,
        go_api_url: str,
        debug: bool = False,
        debug_block_prob: float = 0.15,
        debug_min_delay_s: float = 0.0,
        debug_max_delay_s: float = 0.0,
    ):
        """
        :param go_api_url: Base URL of the Go API (e.g. http://127.0.0.1:8080)
        :param host: Host for this FastAPI server (default: localhost)
        :param port: Port for this FastAPI server (default: 8000)
        """
        self.go_api_url = go_api_url.rstrip("/")
        self.retries = 5
        self.vps = set()
        self.tags = set()
        self.store: MeasurementStore = MeasurementStore()
        self.receiver: MeasurementReceiver = MeasurementReceiver(self.store)
        self.debug = debug
        if not self.debug:
            self.receiver.start_in_background()
        else:
            self.debug_block_prob = debug_block_prob
            self.debug_min_delay_s = debug_min_delay_s
            self.debug_max_delay_s = debug_max_delay_s
            print("Starting measurement API in DEBUG mode...")

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

        # Create Jobs
        jobs = [Job(target, services, vp, "") for vp in vps for target in targets]

        if self.debug:
            self._inject_debug_results(country=country, jobs=jobs)
            return

        self.add_work(jobs)
        return

    def try_get_results(self, country: str) -> List[MeasurementResponse]:
        results = self.store.get_country_batch(country=country)
        # if len(results) > 0:
        #     print(f"Retrieved results for {country}:{results[0].test_url}")
        return self.parse_measurements(results)

    # ---------------------------- Helpers -----------------------------
    def parse_measurements(
        self, results: List[RequestPayload]
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
        # if len(parsed_output) > 0:
        #     print(f"Parsed Output:\n{parsed_output}")
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
        body = {"vantage_points": [{"ip": ip, "services": services} for ip in ips]}
        for ip in ips:
            if ip not in self.vps:
                self.vps.add(ip)
        return self.call_go_api(endpoint, body)

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
        Simulate measurement completion by pushing synthetic RequestPayloads into the store.
        Uses the SAME format parse_measurements() expects.
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

                payload = RequestPayload(
                    vp=j.vantage_point_predicate,
                    location=LocationData(country_name=country, country_code="XX"),
                    service=(j.services[0] if j.services else "https"),
                    test_url=j.keyword,
                    response=[
                        ResponseData(
                            matches_template=(not blocked),
                            start_time=now,
                            end_time=now,
                        )
                    ],
                    anomaly=False,
                    controls_failed=False,
                    stateful_block=blocked,
                )
                # request_id is country
                self.store.record_result(country, payload)

        # run async so orchestrator loop behaves like real life
        threading.Thread(target=worker, daemon=True).start()

    def call_debug_endpoint(self):
        endpoint = "/debug"
        return self.call_go_api(endpoint, method="GET")

    # ---------------------------- HELPERS ----------------------------
    def call_go_api(self, endpoint: str, data: dict = {}, method: str = "POST"):
        """Send a POST request to the Go API and return JSON response."""
        method = method.upper()
        if method not in ["POST", "GET"]:
            print(f"Invalid method: {method}")
            return {"error": "invalid method"}
        retries = 0
        for _ in range(self.retries):
            try:
                url = f"{self.go_api_url}{endpoint}"
                response = requests.post(url, json=data, timeout=10)
                # print(f"Sending message {data}")
                response.raise_for_status()
                return response.json()
            except Exception as e:
                retries += 1
                if retries == self.retries:
                    print(f"[Error] Failed to call Go API: {e}")
                    return {"error": str(e)}
                else:
                    continue


# ---------------------------- MAIN ENTRY ----------------------------
if __name__ == "__main__":
    # api = SmallQuackAPI(
    # go_file="/Users/grahamklingler/Repositories/hyperquackv2/main.go"
    # )
    # print(api.run_go_trial(keyword="google.com", ip="12.47.31.201", port=443))
    api = HyperQuackAPI(go_api_url="http://127.0.0.1:8888")
    vp = "12.47.31.201"
    vps = ["87.190.253.178", vp]
    print(api.add_vantage_points(vps, ["https"]))
    print(
        api.schedule_measurements(
            vps=vps,
            services=["https"],
            targets=["google.com", "example.com", "poop.com"],
        )
    )
    # print(api.add_tags([Tag("mytag", "output.txt", "eval.txt")]))
    # print(api.add_work([Job("google.com", ["http", "echo", "discard", "https"], "*", "")]))
    # api.receiver.debug_start()
