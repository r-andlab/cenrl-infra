"""Receiver / payload validation: a VP with an empty geolocation must not be rejected.

Regression for experiment_five/output4: Hyperquack sends `location: {}` for a VP whose
IP is missing from its geolocation DB. Python required location.country_name, so the
receiver answered 422 to EVERY result from that VP; the aggregator then waited on that
VP's vote forever and froze the whole country (Germany and Turkey, 0 targets finalized).
"""
import json
import logging
import queue
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from Infrastructure.apis.server import MeasurementReceiver, _build_app
from Infrastructure.utils.structures import TestPayload

# a REAL payload from that run (VP 192.214.176.75, location == {})
EMPTY_LOCATION = json.loads(
    (Path(__file__).parent / "fixtures_empty_location_result.json").read_text())


class TestEmptyLocationValidates:
    def test_fixture_really_has_an_empty_location(self):
        assert EMPTY_LOCATION["location"] == {}

    def test_real_payload_with_empty_location_validates(self):
        p = TestPayload(**EMPTY_LOCATION)
        assert p.vp == "192.214.176.75"
        assert p.location.country_name is None and p.location.country_code is None
        assert p.tag == "Germany"

    def test_payload_with_a_full_location_still_validates(self):
        d = dict(EMPTY_LOCATION, location={"country_name": "Germany", "country_code": "DE"})
        p = TestPayload(**d)
        assert p.location.country_name == "Germany" and p.location.country_code == "DE"

    def test_payload_with_no_location_key_validates(self):
        d = {k: v for k, v in EMPTY_LOCATION.items() if k != "location"}
        assert TestPayload(**d).location.country_name is None


class TestReceiverApp:
    def _client(self):
        m, e = queue.Queue(), queue.Queue()
        return TestClient(_build_app(m, e)), m, e

    def test_empty_location_result_is_accepted_and_queued(self):
        client, m, _ = self._client()
        r = client.post("/measurement-done", json=EMPTY_LOCATION)
        assert r.status_code == 200
        assert m.qsize() == 1 and m.get()["vp"] == "192.214.176.75"

    def test_invalid_payload_is_rejected_and_now_logged(self, caplog):
        client, m, _ = self._client()
        bad = {k: v for k, v in EMPTY_LOCATION.items() if k != "test_url"}   # a truly invalid one
        with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
            r = client.post("/measurement-done", json=bad)
        assert r.status_code == 422 and m.qsize() == 0
        assert any("REJECTED payload #1" in rec.getMessage() and "192.214.176.75" in rec.getMessage()
                   for rec in caplog.records), [r.getMessage() for r in caplog.records]

    def test_rejection_logging_is_rate_limited(self, caplog):
        client, _, _ = self._client()
        bad = {k: v for k, v in EMPTY_LOCATION.items() if k != "test_url"}
        with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
            for _ in range(60):
                client.post("/measurement-done", json=bad)
        n = sum("REJECTED payload" in rec.getMessage() for rec in caplog.records)
        assert n == 20                       # first 20 only (500th not reached)


class TestDrainRouting:
    def _receiver(self):
        got = []
        class Store:
            def record_result(self, country, payload):
                got.append((country, payload.vp))
        r = MeasurementReceiver(Store())
        return r, got

    def test_routes_by_tag_even_with_empty_location(self):
        r, got = self._receiver()
        r._measurement_queue.put(TestPayload(**EMPTY_LOCATION).model_dump())
        import time; time.sleep(0.3)         # multiprocessing.Queue feeder thread
        r.drain_queues()
        assert got == [("Germany", "192.214.176.75")]

    def test_no_tag_and_no_country_is_dropped_with_a_warning(self, caplog):
        r, got = self._receiver()
        d = dict(EMPTY_LOCATION, tag=None)
        r._measurement_queue.put(TestPayload(**d).model_dump())
        import time; time.sleep(0.3)
        with caplog.at_level(logging.WARNING):
            r.drain_queues()
        assert got == []
        assert any("no tag and no country" in rec.getMessage() for rec in caplog.records)
