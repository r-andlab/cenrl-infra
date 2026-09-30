"""Tests for HyperQuackAPI: call_go_api GET dispatch and remove_vantage_points()."""

import logging
import pytest
from unittest.mock import MagicMock, patch

from Infrastructure.apis.funneler import HyperQuackAPI


@pytest.fixture
def api():
    """Create a minimal HyperQuackAPI bypassing full __init__."""
    with patch.object(HyperQuackAPI, "__init__", lambda self, *a, **kw: None):
        obj = HyperQuackAPI.__new__(HyperQuackAPI)
    obj.go_api_url = "http://127.0.0.1:8080"
    obj.retries = 5
    obj.vps = {"1.1.1.1", "2.2.2.2"}
    obj.debug = False
    return obj


class TestCallGoApiGetDispatch:
    """Tests for call_go_api method dispatch."""

    @patch("Infrastructure.apis.funneler.requests.get")
    def test_get_calls_requests_get(self, mock_get, api):
        """call_go_api with method='GET' calls requests.get."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = api.call_go_api("/debug", method="GET")

        mock_get.assert_called_once_with("http://127.0.0.1:8080/debug", timeout=10)
        assert result == {"status": "ok"}

    @patch("Infrastructure.apis.funneler.requests.post")
    def test_post_calls_requests_post(self, mock_post, api):
        """call_go_api with method='POST' calls requests.post (unchanged behavior)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        data = {"key": "value"}
        result = api.call_go_api("/endpoint", data=data, method="POST")

        mock_post.assert_called_once_with(
            "http://127.0.0.1:8080/endpoint", json=data, timeout=10
        )
        assert result == {"status": "ok"}


class TestBug1Fix:
    """parse_measurements must use Hyperquack's final anomaly verdict, not
    response[0].matches_template (the first-attempt-only bug)."""

    def _payload(self, anomaly, controls_failed=False):
        p = MagicMock()
        p.test_url = "example.com"
        p.anomaly = anomaly
        p.controls_failed = controls_failed
        return p

    def test_retry_rescued_result_is_not_blocked(self, api):
        results = api.parse_measurements([self._payload(anomaly=False)])
        assert results[0].blocked is False

    def test_genuine_anomaly_is_blocked(self, api):
        results = api.parse_measurements([self._payload(anomaly=True)])
        assert results[0].blocked is True

    def test_controls_failed_result_is_not_blocked(self, api):
        results = api.parse_measurements([self._payload(anomaly=False, controls_failed=True)])
        assert results[0].blocked is False


class TestRemoveVantagePoints:
    """Tests for remove_vantage_points() method."""

    @patch.object(HyperQuackAPI, "call_go_api")
    def test_sends_correct_body(self, mock_call, api):
        """remove_vantage_points sends POST to /remove-vantage-points with correct body."""
        mock_call.return_value = {"unstarted_work": {}}
        api.remove_vantage_points(["1.1.1.1", "2.2.2.2"])

        mock_call.assert_called_once_with(
            "/remove-vantage-points", {"ips": ["1.1.1.1", "2.2.2.2"]}
        )

    @patch.object(HyperQuackAPI, "call_go_api")
    def test_logs_warning_for_unstarted_work(self, mock_call, api, caplog):
        """remove_vantage_points logs warning when response contains non-empty unstarted_work."""
        mock_call.return_value = {
            "unstarted_work": {"1.1.1.1": ["target1.com", "target2.com"]}
        }

        with caplog.at_level(logging.WARNING):
            api.remove_vantage_points(["1.1.1.1"])

        assert "unstarted work" in caplog.text.lower()

    @patch.object(HyperQuackAPI, "call_go_api")
    def test_expect_unstarted_suppresses_warning(self, mock_call, api, caplog):
        """remove_vantage_points with expect_unstarted=True suppresses the warning."""
        mock_call.return_value = {
            "unstarted_work": {"1.1.1.1": ["target1.com", "target2.com"]}
        }

        with caplog.at_level(logging.WARNING):
            api.remove_vantage_points(["1.1.1.1"], expect_unstarted=True)

        assert "unstarted work" not in caplog.text.lower()

    @patch.object(HyperQuackAPI, "call_go_api")
    def test_removes_ips_from_vps_set(self, mock_call, api):
        """remove_vantage_points removes IPs from self.vps set."""
        mock_call.return_value = {"unstarted_work": {}}
        assert "1.1.1.1" in api.vps

        api.remove_vantage_points(["1.1.1.1"])

        assert "1.1.1.1" not in api.vps
        assert "2.2.2.2" in api.vps  # other VP unchanged

    def test_empty_ips_returns_empty(self, api):
        """remove_vantage_points returns {} when ips list is empty."""
        result = api.remove_vantage_points([])
        assert result == {}

    def test_debug_mode_returns_empty(self, api):
        """remove_vantage_points returns {} when self.debug is True."""
        api.debug = True
        result = api.remove_vantage_points(["1.1.1.1"])
        assert result == {}
        # VP should NOT be removed in debug mode
        assert "1.1.1.1" in api.vps


class TestCallGoApiTimeoutAndRetries:
    """call_go_api's timeout/max_tries params -- fixes the retry-storm where
    a slow-but-succeeding add-work call got retried into duplicate queued
    work under the old fixed 10s timeout + 5 retries."""

    @patch("Infrastructure.apis.funneler.requests.post")
    def test_custom_timeout_is_passed_through(self, mock_post, api):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_post.return_value = mock_resp
        api.call_go_api("/endpoint", {}, timeout=600)
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8080/endpoint", json={}, timeout=600,
        )

    @patch("Infrastructure.apis.funneler.requests.post")
    def test_max_tries_one_does_not_retry_on_failure(self, mock_post, api):
        """max_tries=1 must give up after a single failed attempt, not the
        default self.retries=5."""
        mock_post.side_effect = Exception("boom")
        result = api.call_go_api("/endpoint", {}, timeout=900, max_tries=1)
        assert mock_post.call_count == 1
        assert "error" in result

    @patch("Infrastructure.apis.funneler.requests.post")
    def test_default_max_tries_still_uses_self_retries(self, mock_post, api):
        """Omitting max_tries preserves the original behavior bit-for-bit."""
        mock_post.side_effect = Exception("boom")
        api.call_go_api("/endpoint", {})
        assert mock_post.call_count == api.retries


class TestAddWorkAndAddVantagePointsTimeouts:
    """add_work and add_vantage_points each override call_go_api's defaults
    for all-VPs-scale traffic."""

    @patch.object(HyperQuackAPI, "call_go_api")
    def test_add_work_uses_single_long_attempt(self, mock_call, api):
        from Infrastructure.utils.structures import Job, Tag
        mock_call.return_value = {"status": "ok"}
        job = Job("example.com", ["https"], "1.1.1.1", Tag(
            tag="US", result_output_file="r.jsonl", eval_output_file="e.jsonl",
        ))
        api.add_work([job])
        _, kwargs = mock_call.call_args
        assert mock_call.call_args.args[0] == "/add-work"
        assert kwargs.get("timeout") == 900
        assert kwargs.get("max_tries") == 1

    @patch.object(HyperQuackAPI, "call_go_api")
    def test_add_vantage_points_uses_longer_timeout(self, mock_call, api):
        api.vantage_points = None
        mock_call.return_value = {"status": "ok"}
        api.add_vantage_points(["1.1.1.1"], ["https"])
        _, kwargs = mock_call.call_args
        assert kwargs.get("timeout") == 600
