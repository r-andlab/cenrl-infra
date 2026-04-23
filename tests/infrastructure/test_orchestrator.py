import logging
import signal
import time
import unittest
from collections import defaultdict
from datetime import datetime, timezone, timedelta, time as dtime
from unittest.mock import MagicMock, patch, PropertyMock

from Infrastructure.utils.structures import NodeState


class TestRunForeverLoopCondition(unittest.TestCase):
    """Verify run_forever() does not exit when agents dict is empty -- covers RUNT-01/D-05/D-07."""

    def _make_orchestrator(self):
        """Create a minimal Orchestrator with mocked dependencies."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        import threading
        orch.agents = {}
        orch._stop_event = threading.Event()
        orch._next_reset_utc = datetime.now(timezone.utc) + timedelta(days=1)
        orch._draining = False
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = {}
        orch.api = MagicMock()
        orch.api.receiver._process = None  # debug mode
        orch.target_countries = set()
        return orch

    def test_no_exit_on_empty_agents(self):
        """run_forever must NOT exit just because self.agents is empty."""
        orch = self._make_orchestrator()
        tick_count = 0

        def counting_tick():
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 3:
                orch._stop_event.set()  # stop after 3 ticks

        orch.tick = counting_tick
        orch._install_signal_handlers = MagicMock()
        orch._shutdown = MagicMock()

        orch.run_forever()
        self.assertEqual(tick_count, 3)
        orch._shutdown.assert_called_once()


class TestDailyResetTrigger(unittest.TestCase):
    """Verify wall-clock reset fires at midnight UTC -- covers RUNT-02/D-01/D-02."""

    def _make_orchestrator(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        import threading
        orch.agents = {}
        orch._stop_event = threading.Event()
        orch._draining = False
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = {}
        orch.api = MagicMock()
        orch.api.receiver._process = None
        orch.target_countries = set()
        return orch

    def test_daily_reset_triggers(self):
        """When now >= _next_reset_utc and all nodes drained, soft reset fires."""
        orch = self._make_orchestrator()
        # Set next_reset to the past so it triggers immediately
        orch._next_reset_utc = datetime.now(timezone.utc) - timedelta(hours=1)

        node = MagicMock()
        node.in_flight = 0
        node.soft_reset = MagicMock()
        orch.agents = {"US": node}

        tick_count = 0
        def stop_after_one():
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 1:
                orch._stop_event.set()

        orch.tick = stop_after_one
        orch._install_signal_handlers = MagicMock()
        orch._shutdown = MagicMock()

        orch.run_forever()
        node.soft_reset.assert_called_once()

    def test_drain_before_reset(self):
        """When midnight passes but nodes have in-flight, draining is set and no reset yet."""
        orch = self._make_orchestrator()
        orch._next_reset_utc = datetime.now(timezone.utc) - timedelta(hours=1)

        node = MagicMock()
        node.in_flight = 5  # still has in-flight
        node.soft_reset = MagicMock()
        orch.agents = {"US": node}

        tick_count = 0
        def stop_after_one():
            nonlocal tick_count
            tick_count += 1
            if tick_count >= 1:
                orch._stop_event.set()

        orch.tick = stop_after_one
        orch._install_signal_handlers = MagicMock()
        orch._shutdown = MagicMock()

        orch.run_forever()
        self.assertTrue(orch._draining)
        node.soft_reset.assert_not_called()


class TestSignalShutdown(unittest.TestCase):
    """Verify SIGTERM causes clean exit -- covers RUNT-04/D-11."""

    def test_sigterm_shutdown(self):
        """Setting _stop_event causes run_forever to exit and call _shutdown."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        import threading
        orch._stop_event = threading.Event()
        orch._next_reset_utc = datetime.now(timezone.utc) + timedelta(days=1)
        orch._draining = False
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = {}
        orch.api = MagicMock()
        orch.api.receiver._process = None

        node = MagicMock()
        node.write_stats = MagicMock()
        node.in_flight = 0
        orch.agents = {"US": node}
        orch.target_countries = set()

        # Set stop immediately
        orch._stop_event.set()
        orch._install_signal_handlers = MagicMock()

        orch.run_forever()
        node.write_stats.assert_called_once()


class TestDelayedResultLogging(unittest.TestCase):
    """Verify 5-min delayed-result logger -- covers RUNT-05/D-09."""

    def test_delayed_result_logging(self):
        """Warning logged for measurements in-flight longer than 5 minutes."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        # Simulate a measurement scheduled 6 minutes ago
        orch._inflight_times = {"US": {"example.com": time.monotonic() - 360}}
        orch._last_delay_warn = defaultdict(dict)
        orch._check_and_resend_stuck = MagicMock()

        with self.assertLogs(level=logging.WARNING) as cm:
            orch._check_delayed_results()
        self.assertTrue(any("DELAYED" in msg for msg in cm.output))
        self.assertTrue(any("US" in msg for msg in cm.output))
        self.assertTrue(any("example.com" in msg for msg in cm.output))

    def test_no_logging_for_recent_measurements(self):
        """No warning for measurements scheduled less than 5 minutes ago."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        # Simulate a measurement scheduled just now
        orch._inflight_times = {"US": {"example.com": time.monotonic()}}
        orch._last_delay_warn = defaultdict(dict)

        # Should not log
        with self.assertRaises(AssertionError):
            with self.assertLogs(level=logging.WARNING):
                orch._check_delayed_results()


class TestSubprocessHealthCheck(unittest.TestCase):
    """Verify subprocess crash detection -- covers RUNT-06/D-12."""

    def test_subprocess_health_check_dead(self):
        """Dead subprocess returns False."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.api = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 1
        orch.api.receiver._process = proc
        self.assertFalse(orch._subprocess_healthy())

    def test_subprocess_health_check_alive(self):
        """Alive subprocess returns True."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.api = MagicMock()
        proc = MagicMock()
        proc.is_alive.return_value = True
        orch.api.receiver._process = proc
        self.assertTrue(orch._subprocess_healthy())

    def test_subprocess_health_check_debug_mode(self):
        """In debug mode (_process is None), returns True."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.api = MagicMock()
        orch.api.receiver._process = None
        self.assertTrue(orch._subprocess_healthy())


class _OrchestratorTestHelper:
    """Shared helper for creating a minimal Orchestrator with mocked dependencies."""

    @staticmethod
    def make_orchestrator():
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        import threading
        from collections import defaultdict
        orch.agents = {}
        orch._stop_event = threading.Event()
        orch._next_reset_utc = datetime.now(timezone.utc) + timedelta(days=1)
        orch._draining = False
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = defaultdict(dict)
        orch._vp_failure_counts = defaultdict(int)
        orch.api = MagicMock()
        orch.api.receiver._process = None
        orch.api.debug = False
        orch.target_countries = set()
        orch.services = ["https"]
        orch.vantage_points = MagicMock()
        orch.completed_agents = set()
        orch.eval_store = MagicMock()
        return orch


class TestRegisterTargetsWiring(unittest.TestCase):
    """Verify tick() calls aggregator.register_targets, not update_aggregator_vps."""

    def test_tick_calls_register_targets(self):
        """tick() should call aggregator.register_targets with country, targets, and VP set."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        node = MagicMock()
        node.state = NodeState.IDLE
        node.maybe_request_more.return_value = ["example.com", "test.org"]
        orch.agents = {"US": node}
        orch.target_countries = {"US"}
        orch.vantage_points.get_active.return_value = ["1.1.1.1", "2.2.2.2"]
        orch.vantage_points.get_inactive.return_value = []

        orch.tick()

        orch.api.aggregator.register_targets.assert_called_once_with(
            "US", ["example.com", "test.org"], {"1.1.1.1", "2.2.2.2"}
        )
        # update_aggregator_vps should NOT be called
        orch.api.update_aggregator_vps.assert_not_called()

    def test_tick_skips_scheduling_when_draining(self):
        """When draining, tick() should not call register_targets."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch._draining = True
        node = MagicMock()
        node.state = NodeState.IDLE
        orch.agents = {"US": node}
        orch.target_countries = {"US"}

        orch.tick()

        orch.api.aggregator.register_targets.assert_not_called()


class TestVPRemovalOnEvalFailure(unittest.TestCase):
    """Verify _process_eval_results calls remove_vantage_points with failed VPs."""

    def test_batch_remove_failed_vps(self):
        """Failed eval VPs should be batch-removed from Hyperquack."""
        orch = _OrchestratorTestHelper.make_orchestrator()

        # Create two failed eval payloads
        payload1 = MagicMock()
        payload1.vp = "1.1.1.1"
        payload1.template = None
        payload2 = MagicMock()
        payload2.vp = "2.2.2.2"
        payload2.template = None

        orch.eval_store.drain.return_value = [payload1, payload2]
        orch.eval_store.get_country.side_effect = lambda vp: "US"
        orch.vantage_points.reject_vp.return_value = None
        orch.vantage_points.get_active.return_value = []
        orch.vantage_points.get_inactive.return_value = []

        orch._process_eval_results()

        orch.api.remove_vantage_points.assert_called_once_with(["1.1.1.1", "2.2.2.2"])

    def test_no_remove_when_all_pass(self):
        """If all VPs pass evaluation, remove_vantage_points should not be called."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        # Pre-create the node so _process_eval_results doesn't try to construct one
        orch.agents = {"US": MagicMock()}

        payload = MagicMock()
        payload.vp = "1.1.1.1"
        payload.template = "some_template"

        orch.eval_store.drain.return_value = [payload]
        orch.eval_store.get_country.return_value = "US"

        orch._process_eval_results()

        orch.api.remove_vantage_points.assert_not_called()


class TestVPRemovalOnHealthCheck(unittest.TestCase):
    """Verify _check_vp_health calls remove_vantage_points for dropped VPs."""

    def test_remove_vp_on_threshold(self):
        """When VP exceeds failure threshold, remove_vantage_points is called."""
        from Infrastructure.main.orchestrator import VP_FAILURE_THRESHOLD
        orch = _OrchestratorTestHelper.make_orchestrator()

        # Pre-set failure count just below threshold
        orch._vp_failure_counts[("US", "1.1.1.1")] = VP_FAILURE_THRESHOLD - 1

        result = MagicMock()
        result.vp = "1.1.1.1"
        result.controls_failed = True

        orch.vantage_points.reject_vp.return_value = None

        orch._check_vp_health("US", [result])

        orch.api.remove_vantage_points.assert_called_once_with(["1.1.1.1"])

    def test_no_remove_below_threshold(self):
        """Below threshold, remove_vantage_points should not be called."""
        orch = _OrchestratorTestHelper.make_orchestrator()

        result = MagicMock()
        result.vp = "1.1.1.1"
        result.controls_failed = True

        orch._check_vp_health("US", [result])

        orch.api.remove_vantage_points.assert_not_called()


class TestCheckAndResendStuck(unittest.TestCase):
    """Test /debug poll and resend logic for stuck measurements."""

    def test_debug_mode_noop(self):
        """_check_and_resend_stuck should be a no-op in debug mode."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.api.debug = True

        orch._check_and_resend_stuck({"US": ["example.com"]})

        orch.api.call_debug_endpoint.assert_not_called()

    def test_single_debug_call_per_cycle(self):
        """Only one /debug call regardless of number of stuck targets/countries."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.api.call_debug_endpoint.return_value = []
        orch.api.aggregator._lock = MagicMock()
        orch.api.aggregator._target_expected = {}
        orch.api.aggregator._pending = {}

        orch._check_and_resend_stuck({"US": ["a.com", "b.com"], "UK": ["c.com"]})

        orch.api.call_debug_endpoint.assert_called_once()

    def test_non_list_response_handled(self):
        """Non-list /debug response should log warning and return."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.api.call_debug_endpoint.return_value = {"error": "something"}

        with self.assertLogs(level=logging.WARNING) as cm:
            orch._check_and_resend_stuck({"US": ["example.com"]})

        self.assertTrue(any("Unexpected /debug response" in msg for msg in cm.output))
        orch.api.schedule_measurements.assert_not_called()

    def test_skip_when_in_unstarted_work(self):
        """If VP still has target in unstarted_work, do not resend."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.api.call_debug_endpoint.return_value = [
            {
                "ip": "1.1.1.1",
                "pending_removal": False,
                "unstarted_work": [{"keyword": "example.com", "service": "https"}],
            }
        ]
        orch.api.aggregator._lock = MagicMock()
        orch.api.aggregator._target_expected = {
            ("US", "example.com"): frozenset(["1.1.1.1"])
        }
        orch.api.aggregator._pending = {("US", "example.com"): {}}

        orch.vantage_points.get_active.return_value = ["1.1.1.1"]

        orch._check_and_resend_stuck({"US": ["example.com"]})

        orch.api.schedule_measurements.assert_not_called()

    def test_resend_to_original_vp(self):
        """If VP lost the measurement and is not pending_removal, resend to same VP."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.api.call_debug_endpoint.return_value = [
            {
                "ip": "1.1.1.1",
                "pending_removal": False,
                "unstarted_work": [],  # target NOT in unstarted_work => lost
            }
        ]
        orch.api.aggregator._lock = MagicMock()
        orch.api.aggregator._target_expected = {
            ("US", "example.com"): frozenset(["1.1.1.1"])
        }
        orch.api.aggregator._pending = {("US", "example.com"): {}}

        orch.vantage_points.get_active.return_value = ["1.1.1.1"]

        orch._check_and_resend_stuck({"US": ["example.com"]})

        orch.api.schedule_measurements.assert_called_once_with(
            vps=["1.1.1.1"],
            services=["https"],
            targets=["example.com"],
            country="US",
        )

    def test_resend_to_alternative_when_pending_removal(self):
        """If original VP is pending_removal, resend to an alternative VP."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.api.call_debug_endpoint.return_value = [
            {
                "ip": "1.1.1.1",
                "pending_removal": True,
                "unstarted_work": [],  # lost
            }
        ]
        orch.api.aggregator._lock = MagicMock()
        orch.api.aggregator._target_expected = {
            ("US", "example.com"): frozenset(["1.1.1.1"])
        }
        orch.api.aggregator._pending = {("US", "example.com"): {}}

        orch.vantage_points.get_active.return_value = ["1.1.1.1", "3.3.3.3"]

        orch._check_and_resend_stuck({"US": ["example.com"]})

        orch.api.schedule_measurements.assert_called_once_with(
            vps=["3.3.3.3"],
            services=["https"],
            targets=["example.com"],
            country="US",
        )

    def test_delayed_results_collects_stuck_and_calls_resend(self):
        """_check_delayed_results should collect stuck targets and call _check_and_resend_stuck."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        # Simulate a measurement stuck for 6 minutes
        orch._inflight_times = {"US": {"example.com": time.monotonic() - 360}}
        orch._check_and_resend_stuck = MagicMock()

        orch._check_delayed_results()

        orch._check_and_resend_stuck.assert_called_once()
        call_args = orch._check_and_resend_stuck.call_args[0][0]
        self.assertIn("US", call_args)
        self.assertIn("example.com", call_args["US"])


if __name__ == "__main__":
    unittest.main()
