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
