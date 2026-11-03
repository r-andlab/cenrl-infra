import csv as _csv_module
import logging
import signal
import threading
import time
import unittest
from collections import defaultdict
from datetime import datetime, timezone, timedelta, time as dtime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call

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
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = {}
        orch._last_checkpoint_time = time.monotonic()
        orch._checkpoint_interval = 3600.0
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
    """Verify midnight UTC trigger fires save_daily ONLY -- Phase 02.1 D-09 / D-10.

    Under Phase 02.1, midnight no longer drains or calls soft_reset. Per-country
    soft resets are measurement-count-driven inside RegionalNode. This class
    verifies the gutted save-only handler in run_forever().
    """

    def _make_orchestrator(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        import threading
        orch.agents = {}
        orch._stop_event = threading.Event()
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = {}
        orch._last_checkpoint_time = time.monotonic()
        orch._checkpoint_interval = 3600.0
        orch.output_folder = "/tmp/test_orchestrator_output"
        orch.api = MagicMock()
        orch.api.receiver._process = None
        orch.target_countries = set()
        return orch

    def test_midnight_triggers_save_daily_only(self):
        """At midnight UTC, run_forever fires save_daily for each node and advances
        _next_reset_utc by 1 day. soft_reset is NOT called -- that is per-country
        and measurement-count-driven (Phase 02.1 D-09)."""
        orch = self._make_orchestrator()
        node = MagicMock()
        orch.agents = {"China": node}
        # Force midnight to be in the past so the handler fires
        before = datetime.now(timezone.utc) - timedelta(seconds=1)
        orch._next_reset_utc = before
        # Stub helpers that the run_forever loop relies on
        orch._state_dir_for = MagicMock(return_value=Path("/tmp/state"))
        orch._subprocess_healthy = MagicMock(return_value=True)
        orch.tick = MagicMock()
        orch._check_delayed_results = MagicMock()
        orch._checkpoint_all_nodes = MagicMock()

        # Stop after one iteration
        def stop_after_first(*a, **kw):
            orch._stop_event.set()
        orch.tick.side_effect = stop_after_first
        orch._install_signal_handlers = MagicMock()
        orch._shutdown = MagicMock()

        orch.run_forever()

        node.save_daily.assert_called_once()
        args = node.save_daily.call_args[0]
        self.assertRegex(args[0], r"^\d{4}-\d{2}-\d{2}$")
        # state_dir is the second positional arg
        self.assertEqual(args[1], Path("/tmp/state"))
        node.soft_reset.assert_not_called()
        # _next_reset_utc advanced by exactly 1 day from the original
        self.assertEqual(orch._next_reset_utc, before + timedelta(days=1))


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
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = defaultdict(dict)
        orch._vp_failure_counts = defaultdict(int)
        # Phase 4 D-02: cumulative (success, fail) ledger for WEIGHTED_VOTE.
        orch._vp_outcomes = defaultdict(lambda: [0, 0])
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

        # Phase 3 D-04: register_targets now also receives a schedule_times
        # kwarg from the orchestrator. Use ANY to avoid coupling this test
        # to the exact monotonic value picked at scheduling.
        from unittest.mock import ANY
        orch.api.aggregator.register_targets.assert_called_once_with(
            "US", ["example.com", "test.org"], {"1.1.1.1", "2.2.2.2"},
            schedule_times=ANY,
        )
        # update_aggregator_vps should NOT be called
        orch.api.update_aggregator_vps.assert_not_called()

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

        orch.api.remove_vantage_points.assert_called_once_with(["1.1.1.1", "2.2.2.2"], expect_unstarted=True)

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

        orch.api.remove_vantage_points.assert_called_once_with(["1.1.1.1"], expect_unstarted=True)

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
        # WR-03: orchestrator now reads aggregator pending state via the
        # public snapshot_pending() API instead of touching private attrs.
        orch.api.aggregator.snapshot_pending = MagicMock(return_value={})

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
        # WR-03: snapshot_pending returns {target: (expected, received)}
        orch.api.aggregator.snapshot_pending = MagicMock(return_value={
            "example.com": (frozenset(["1.1.1.1"]), frozenset()),
        })

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
        # WR-03: snapshot_pending returns {target: (expected, received)}
        orch.api.aggregator.snapshot_pending = MagicMock(return_value={
            "example.com": (frozenset(["1.1.1.1"]), frozenset()),
        })

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
        # WR-03: snapshot_pending returns {target: (expected, received)}
        orch.api.aggregator.snapshot_pending = MagicMock(return_value={
            "example.com": (frozenset(["1.1.1.1"]), frozenset()),
        })

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


class TestStateDirFor(unittest.TestCase):
    """Verify _state_dir_for() path construction for various country names (Plan 02-03)."""

    def _orch(self, output_folder: str):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.output_folder = output_folder
        return orch

    def test_simple_country_name(self):
        orch = self._orch("/tmp/out")
        self.assertEqual(orch._state_dir_for("Germany"), Path("/tmp/out/Germany/state"))

    def test_country_name_with_space_replaced(self):
        orch = self._orch("/tmp/out")
        self.assertEqual(
            orch._state_dir_for("United States"),
            Path("/tmp/out/United_States/state"),
        )

    def test_returns_path_instance(self):
        orch = self._orch("/tmp/out")
        result = orch._state_dir_for("Germany")
        self.assertIsInstance(result, Path)


class TestCheckpointAllNodes(unittest.TestCase):
    """Verify _checkpoint_all_nodes() iterates all agents and isolates failures."""

    def _orch_with_agents(self, agents):
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.output_folder = "/tmp/out"
        orch.agents = agents
        return orch

    def test_calls_save_checkpoint_on_each_agent(self):
        node_de = MagicMock()
        node_us = MagicMock()
        orch = self._orch_with_agents({"Germany": node_de, "United States": node_us})

        orch._checkpoint_all_nodes()

        node_de.save_checkpoint.assert_called_once_with(
            Path("/tmp/out/Germany/state")
        )
        node_us.save_checkpoint.assert_called_once_with(
            Path("/tmp/out/United_States/state")
        )

    def test_continues_after_node_save_raises(self):
        bad_node = MagicMock()
        bad_node.save_checkpoint.side_effect = RuntimeError("disk full")
        good_node = MagicMock()
        # Order matters: bad first, good second — must still save the good one.
        orch = self._orch_with_agents({"Bad": bad_node, "Good": good_node})

        with self.assertLogs(level=logging.ERROR) as cm:
            orch._checkpoint_all_nodes()

        bad_node.save_checkpoint.assert_called_once()
        good_node.save_checkpoint.assert_called_once()
        self.assertTrue(any("Bad" in m for m in cm.output))


class TestHourlyCheckpointTimer(unittest.TestCase):
    """Verify the monotonic-time hourly checkpoint guard in run_forever()."""

    def _orch(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        import threading
        orch.agents = {}
        orch._stop_event = threading.Event()
        orch._next_reset_utc = datetime.now(timezone.utc) + timedelta(days=1)
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = {}
        orch._checkpoint_interval = 3600.0
        orch.api = MagicMock()
        orch.api.receiver._process = None
        orch.target_countries = set()
        orch._install_signal_handlers = MagicMock()
        orch._shutdown = MagicMock()
        orch._checkpoint_all_nodes = MagicMock()
        orch.tick = MagicMock()
        return orch

    def test_checkpoint_fires_after_interval_elapsed(self):
        """When monotonic clock has advanced past _checkpoint_interval, the
        checkpoint method should be invoked exactly once on the next loop pass."""
        orch = self._orch()
        # Pretend the last checkpoint happened more than 1 hour ago.
        orch._last_checkpoint_time = time.monotonic() - 3700.0

        # Run a single iteration of run_forever.
        def stop_after_one(*_a, **_kw):
            orch._stop_event.set()
        orch.tick.side_effect = stop_after_one

        orch.run_forever()

        orch._checkpoint_all_nodes.assert_called_once()
        # And the timer was advanced (i.e. _last_checkpoint_time updated).
        self.assertGreater(orch._last_checkpoint_time, time.monotonic() - 60.0)

    def test_checkpoint_does_not_fire_before_interval(self):
        """If less than _checkpoint_interval has elapsed since last checkpoint,
        _checkpoint_all_nodes must NOT be called."""
        orch = self._orch()
        # Last checkpoint was 5 seconds ago — well under 3600s.
        orch._last_checkpoint_time = time.monotonic() - 5.0

        def stop_after_one(*_a, **_kw):
            orch._stop_event.set()
        orch.tick.side_effect = stop_after_one

        orch.run_forever()

        orch._checkpoint_all_nodes.assert_not_called()


class TestShutdownSavesCheckpoint(unittest.TestCase):
    """Verify _shutdown() calls write_stats AND save_checkpoint for each node."""

    def test_shutdown_calls_both_write_stats_and_save_checkpoint(self):
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.output_folder = "/tmp/out"
        node = MagicMock()
        orch.agents = {"Germany": node}

        orch._shutdown()

        node.write_stats.assert_called_once()
        node.save_checkpoint.assert_called_once_with(Path("/tmp/out/Germany/state"))

    def test_shutdown_save_checkpoint_runs_after_write_stats(self):
        """write_stats must be invoked before save_checkpoint per task action step 5."""
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.output_folder = "/tmp/out"
        node = MagicMock()
        orch.agents = {"Germany": node}

        orch._shutdown()

        method_names = [c[0] for c in node.method_calls]
        ws_idx = method_names.index("write_stats")
        sc_idx = method_names.index("save_checkpoint")
        self.assertLess(ws_idx, sc_idx, "write_stats must be called before save_checkpoint")

    def test_shutdown_continues_when_one_node_save_raises(self):
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.output_folder = "/tmp/out"
        bad = MagicMock()
        bad.save_checkpoint.side_effect = OSError("disk full")
        good = MagicMock()
        orch.agents = {"Bad": bad, "Good": good}

        with self.assertLogs(level=logging.ERROR):
            orch._shutdown()

        good.save_checkpoint.assert_called_once_with(Path("/tmp/out/Good/state"))


class TestMidnightSaveDailyLoop(unittest.TestCase):
    """Phase 02.1 D-09 / D-10: midnight handler in run_forever calls save_daily for
    every active node, advances _next_reset_utc by 1 day, and NEVER calls soft_reset.

    This class replaces the legacy TestDailySaveBeforeReset which tested the now-removed
    Orchestrator._perform_soft_reset method. Save-before-reset ordering no longer applies
    at midnight because midnight does not trigger soft_reset under Phase 02.1.
    """

    def _orch_with_agents(self, agents):
        orch = _OrchestratorTestHelper.make_orchestrator()
        orch.output_folder = "/tmp/out"
        orch._last_checkpoint_time = time.monotonic()
        orch._checkpoint_interval = 3600.0
        orch.agents = agents
        # Force midnight handler to fire on first iteration
        orch._next_reset_utc = datetime.now(timezone.utc) - timedelta(seconds=1)
        orch._subprocess_healthy = MagicMock(return_value=True)
        orch._check_delayed_results = MagicMock()
        orch._checkpoint_all_nodes = MagicMock()
        orch._install_signal_handlers = MagicMock()
        orch._shutdown = MagicMock()
        orch.tick = MagicMock(side_effect=lambda *a, **kw: orch._stop_event.set())
        return orch

    def test_midnight_calls_save_daily_for_each_node(self):
        """Each active node receives one save_daily call with (date_str, state_dir)."""
        node_de = MagicMock()
        node_us = MagicMock()
        orch = self._orch_with_agents({"Germany": node_de, "United States": node_us})

        orch.run_forever()

        node_de.save_daily.assert_called_once()
        node_us.save_daily.assert_called_once()
        # state_dir must be the per-country path
        de_args = node_de.save_daily.call_args[0]
        us_args = node_us.save_daily.call_args[0]
        self.assertEqual(de_args[1], Path("/tmp/out/Germany/state"))
        self.assertEqual(us_args[1], Path("/tmp/out/United_States/state"))

    def test_midnight_save_daily_uses_utc_date_string(self):
        """date_str must parse as today's UTC date in YYYY-MM-DD form."""
        node = MagicMock()
        orch = self._orch_with_agents({"Germany": node})

        orch.run_forever()

        args = node.save_daily.call_args[0]
        date_str = args[0]
        parsed = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        today_utc = datetime.now(timezone.utc).date()
        self.assertEqual(parsed.date(), today_utc)

    def test_midnight_does_not_call_soft_reset(self):
        """soft_reset must NOT be called at midnight (Phase 02.1 D-09)."""
        node = MagicMock()
        orch = self._orch_with_agents({"Germany": node})

        orch.run_forever()

        node.soft_reset.assert_not_called()

    def test_midnight_continues_when_save_daily_raises(self):
        """T-02.1-04: a failing save_daily must NOT block other nodes' saves."""
        bad = MagicMock()
        bad.save_daily.side_effect = OSError("disk full")
        good = MagicMock()
        # Iteration order matters: bad first, then good.
        orch = self._orch_with_agents({"Bad": bad, "Good": good})

        with self.assertLogs(level=logging.ERROR):
            orch.run_forever()

        bad.save_daily.assert_called_once()
        good.save_daily.assert_called_once()


class TestPerCountryResetCallback(unittest.TestCase):
    """Tests for Orchestrator._on_country_reset (Phase 02.1 D-12)."""

    def _make_orchestrator(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        from collections import defaultdict
        orch._inflight_times = defaultdict(dict)
        orch._last_delay_warn = defaultdict(dict)
        return orch

    def test_clears_only_target_country_inflight_times(self):
        """_on_country_reset('China') clears China inflight_times only -- Russia stays."""
        orch = self._make_orchestrator()
        orch._inflight_times["China"]["target1"] = 100.0
        orch._inflight_times["Russia"]["target2"] = 200.0

        orch._on_country_reset("China")

        self.assertNotIn("China", orch._inflight_times)
        self.assertIn("Russia", orch._inflight_times)
        self.assertEqual(orch._inflight_times["Russia"]["target2"], 200.0)

    def test_clears_only_target_country_last_delay_warn(self):
        """_on_country_reset('China') clears China last_delay_warn only -- Russia stays."""
        orch = self._make_orchestrator()
        now = datetime.now(timezone.utc)
        orch._last_delay_warn["China"]["target1"] = now
        orch._last_delay_warn["Russia"]["target2"] = now

        orch._on_country_reset("China")

        self.assertNotIn("China", orch._last_delay_warn)
        self.assertIn("Russia", orch._last_delay_warn)

    def test_unknown_country_no_error(self):
        """_on_country_reset on a country with no entries does not raise."""
        orch = self._make_orchestrator()
        try:
            orch._on_country_reset("NonExistent")
        except Exception as exc:
            self.fail(f"_on_country_reset must not raise on unknown country, got: {exc}")

    def test_clearing_one_country_independence(self):
        """Two countries on different iterations: clearing China leaves Russia entries untouched.

        Direct invariant from CONTEXT.md: 'one country could be on its fifth iteration while
        another is on the second' -- both inflight tracking dicts and last_delay_warn dicts
        remain country-isolated.
        """
        orch = self._make_orchestrator()
        orch._inflight_times["China"]["t1"] = 1.0
        orch._inflight_times["Russia"]["t2"] = 2.0
        orch._inflight_times["Russia"]["t3"] = 3.0
        orch._last_delay_warn["Russia"]["t2"] = datetime.now(timezone.utc)

        orch._on_country_reset("China")

        # Russia full state preserved
        self.assertEqual(orch._inflight_times["Russia"], {"t2": 2.0, "t3": 3.0})
        self.assertIn("t2", orch._last_delay_warn["Russia"])


class TestCrossCountryIndependence(unittest.TestCase):
    """End-to-end invariant: per-country reset cadences do not couple (Phase 02.1 D-02, D-09, D-12)."""

    def _make_orchestrator(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch._inflight_times = defaultdict(dict)
        orch._last_delay_warn = defaultdict(dict)
        return orch

    def test_two_countries_diverge_in_episode_idx_without_blocking(self):
        """Simulate China hitting quota repeatedly while Russia drains slowly.

        After several rounds of China hitting quota, China's episode_idx should
        be > Russia's episode_idx, and Russia's _inflight_times should NOT have
        been cleared by China's resets (which is what _on_country_reset prevents).
        """
        orch = self._make_orchestrator()

        china_node = MagicMock()
        china_node.country = "China"
        china_node.episode_idx = 1

        russia_node = MagicMock()
        russia_node.country = "Russia"
        russia_node.episode_idx = 1

        orch.agents = {"China": china_node, "Russia": russia_node}
        orch._inflight_times["China"]["t1"] = 1.0
        orch._inflight_times["Russia"]["t10"] = 10.0
        orch._last_delay_warn["Russia"]["t10"] = datetime.now(timezone.utc)

        # China resets 4 times (simulating 4 iteration boundaries)
        for _ in range(4):
            china_node.episode_idx += 1
            orch._on_country_reset("China")
            # Re-add an inflight entry to simulate next iteration scheduling
            orch._inflight_times["China"]["t1"] = 5.0

        # China is on iteration 5; Russia still on iteration 1
        self.assertEqual(china_node.episode_idx, 5)
        self.assertEqual(russia_node.episode_idx, 1)

        # Russia's tracking is undisturbed
        self.assertEqual(orch._inflight_times["Russia"], {"t10": 10.0})
        self.assertIn("t10", orch._last_delay_warn["Russia"])


class TestShutdownClosesMeasurementsWriters(unittest.TestCase):
    """Phase 3 D-01: Orchestrator._shutdown must call close_measurements on every active node."""

    def test_shutdown_calls_close_measurements_per_node(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.output_folder = "/tmp/x"
        node_us = MagicMock()
        node_de = MagicMock()
        orch.agents = {"US": node_us, "Germany": node_de}
        orch._state_dir_for = MagicMock(return_value=Path("/tmp/x/state"))
        orch._shutdown()
        node_us.close_measurements.assert_called_once()
        node_de.close_measurements.assert_called_once()

    def test_shutdown_continues_when_one_node_close_raises(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.output_folder = "/tmp/x"
        bad_node = MagicMock()
        bad_node.close_measurements.side_effect = OSError("disk full")
        good_node = MagicMock()
        orch.agents = {"BadCountry": bad_node, "GoodCountry": good_node}
        orch._state_dir_for = MagicMock(return_value=Path("/tmp/x/state"))
        # Must not raise
        with self.assertLogs(level=logging.ERROR):
            orch._shutdown()
        good_node.close_measurements.assert_called_once()
class TestTickReturnsTimings(unittest.TestCase):
    """Phase 3 D-07/D-08/D-11: tick() returns a six-field timing dict."""

    def _make_orchestrator(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.agents = {}
        orch.completed_agents = set()
        orch.target_countries = set()
        orch._inflight_times = defaultdict(dict)
        orch._last_delay_warn = defaultdict(dict)
        orch.api = MagicMock()
        orch.api.receiver.drain_queues = MagicMock()
        orch.api.aggregator.get_ready = MagicMock(return_value=[])
        orch.api.drain_raw_results = MagicMock(return_value=[])
        orch._process_eval_results = MagicMock()
        orch.vantage_points = MagicMock()
        orch.services = ["https"]
        return orch

    def test_tick_returns_six_step_timings(self):
        orch = self._make_orchestrator()
        result = orch.tick()
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "drain", "eval_processing", "aggregate",
            "model_update", "schedule", "total",
        }
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"
            assert v >= 0.0, f"{k} is negative: {v}"

    def test_tick_total_geq_outer_steps(self):
        orch = self._make_orchestrator()
        result = orch.tick()
        # total bounds drain + eval_processing (outer-most steps)
        assert result["total"] >= result["drain"] + result["eval_processing"] - 1e-6

    def test_tick_per_country_steps_zero_with_no_agents(self):
        orch = self._make_orchestrator()
        result = orch.tick()
        assert result["aggregate"] == 0.0
        assert result["model_update"] == 0.0
        assert result["schedule"] == 0.0


class TestTickTimingsCsv(unittest.TestCase):
    """Phase 3 D-07/D-09/D-10: tick_timings.csv schema and flush cadence."""

    def _make_orchestrator(self, tmp_path):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.output_folder = str(tmp_path)
        orch.agents = {}
        orch.completed_agents = set()
        orch.target_countries = set()
        orch._stop_event = threading.Event()
        orch._next_reset_utc = datetime.now(timezone.utc) + timedelta(days=1)
        orch._last_delay_warn = defaultdict(dict)
        orch._inflight_times = defaultdict(dict)
        orch._last_checkpoint_time = time.monotonic()
        orch._checkpoint_interval = 3600.0
        orch.api = MagicMock()
        orch.api.receiver._process = None
        orch._install_signal_handlers = MagicMock()
        orch._subprocess_healthy = MagicMock(return_value=True)
        orch._check_delayed_results = MagicMock()
        orch._checkpoint_all_nodes = MagicMock()
        orch._state_dir_for = MagicMock(return_value=Path(str(tmp_path)) / "state")
        orch._shutdown = MagicMock()
        return orch

    def test_tick_timings_csv_header_and_rows(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        tick_count = [0]

        def fake_tick():
            tick_count[0] += 1
            if tick_count[0] >= 3:
                orch._stop_event.set()
            return {
                "drain": 0.001, "eval_processing": 0.002,
                "aggregate": 0.003, "model_update": 0.004,
                "schedule": 0.005, "total": 0.015,
            }
        orch.tick = fake_tick

        # Patch sleep to no-op to keep the test fast
        with patch("Infrastructure.main.orchestrator.sleep", lambda _x: None):
            orch.run_forever()

        csv_path = tmp / "tick_timings.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "tick_idx,utc_timestamp,n_active_countries,drain,eval_processing,aggregate,model_update,schedule,total"
        # 1 header + 3 data rows
        assert len(lines) == 4

    def test_tick_timings_csv_one_row_per_tick(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        tick_count = [0]

        def fake_tick():
            tick_count[0] += 1
            if tick_count[0] >= 5:
                orch._stop_event.set()
            return {"drain": 0.0, "eval_processing": 0.0, "aggregate": 0.0,
                    "model_update": 0.0, "schedule": 0.0, "total": 0.0}
        orch.tick = fake_tick

        with patch("Infrastructure.main.orchestrator.sleep", lambda _x: None):
            orch.run_forever()

        with open(tmp / "tick_timings.csv") as f:
            rows = list(_csv_module.DictReader(f))
        assert len(rows) == 5
        # tick_idx is monotonic 1..5
        assert [int(r["tick_idx"]) for r in rows] == [1, 2, 3, 4, 5]

    def test_tick_timings_csv_flushed_on_shutdown(self):
        """The handle is flushed+closed BEFORE _shutdown so the file is readable post-exit."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)

        def fake_tick():
            orch._stop_event.set()  # exit after one tick
            return {"drain": 0.0, "eval_processing": 0.0, "aggregate": 0.0,
                    "model_update": 0.0, "schedule": 0.0, "total": 0.0}
        orch.tick = fake_tick

        with patch("Infrastructure.main.orchestrator.sleep", lambda _x: None):
            orch.run_forever()

        # File is readable post-exit and has 1 data row
        with open(tmp / "tick_timings.csv") as f:
            rows = list(_csv_module.DictReader(f))
        assert len(rows) == 1
        orch._shutdown.assert_called_once()


class TestRunConfigSnapshot(unittest.TestCase):
    """Phase 3 D-12/D-13/D-14: run_config.json snapshot at startup."""

    def _make_orchestrator(self, tmp_path, params=None, vp_pool_config=None):
        """Construct a partially-initialized Orchestrator with the state
        _write_run_config needs, bypassing the full __init__ flow."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.output_folder = str(tmp_path)
        orch.params = params or {"action_space_file": "inputs/foo.csv", "m": 1000}
        orch._go_api_endpoint = "http://127.0.0.1:8888"
        orch.services = ["https"]
        orch.vps_per_country = 3
        orch._vp_pool_config = vp_pool_config or {
            "ev_file": "local/ev-certs.csv",
            "vp_pool_file": "local/vp_pool.csv",
            "blocklist_file": "local/blocklist.txt",
            "max_countries": 15,
            "blocked_countries": [],
        }
        orch.target_countries = {"US", "Germany"}
        orch.api = MagicMock()
        orch.api.debug = False
        return orch

    def test_run_config_json_written(self):
        import tempfile, json as _json
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        orch._write_run_config()
        config_path = tmp / "run_config.json"
        assert config_path.exists()
        with open(config_path) as f:
            data = _json.load(f)
        assert set(data.keys()) == {"cli_params", "hyperquack", "vp_pool_inputs", "provenance"}

    def test_run_config_hyperquack_group(self):
        import tempfile, json as _json
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        orch._write_run_config()
        with open(tmp / "run_config.json") as f:
            data = _json.load(f)
        assert data["hyperquack"]["go_api_endpoint"] == "http://127.0.0.1:8888"
        assert data["hyperquack"]["services"] == ["https"]
        assert data["hyperquack"]["vps_per_country"] == 3
        assert data["hyperquack"]["debug"] is False

    def test_run_config_vp_pool_inputs_group(self):
        import tempfile, json as _json
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        orch._write_run_config()
        with open(tmp / "run_config.json") as f:
            data = _json.load(f)
        vpi = data["vp_pool_inputs"]
        assert vpi["ev_file"] == "local/ev-certs.csv"
        assert vpi["vp_pool_file"] == "local/vp_pool.csv"
        assert vpi["blocklist_file"] == "local/blocklist.txt"
        assert vpi["max_countries"] == 15
        assert vpi["blocked_countries"] == []

    def test_run_config_provenance_group(self):
        import tempfile, json as _json
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        orch._write_run_config()
        with open(tmp / "run_config.json") as f:
            data = _json.load(f)
        prov = data["provenance"]
        assert sorted(prov["target_countries"]) == ["Germany", "US"]
        assert prov["action_space_file"] == "inputs/foo.csv"
        # git fields: either real values or fallback (null + "unknown")
        assert "git_sha" in prov
        assert prov["git_dirty"] in ("clean", "dirty", "unknown")
        assert "start_utc_timestamp" in prov
        assert "python_version" in prov

    def test_run_config_git_fallback_in_non_git_env(self):
        """Per D-13: subprocess failure must NOT abort startup."""
        import tempfile, json as _json
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        with patch("Infrastructure.main.orchestrator.subprocess.check_output",
                   side_effect=FileNotFoundError("git not on PATH")):
            orch._write_run_config()  # must NOT raise
        with open(tmp / "run_config.json") as f:
            data = _json.load(f)
        assert data["provenance"]["git_sha"] is None
        assert data["provenance"]["git_dirty"] == "unknown"

    def test_run_config_emits_info_log(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        with self.assertLogs("Infrastructure.main.orchestrator", level="INFO") as cm:
            orch._write_run_config()
        joined = "\n".join(cm.output)
        assert "Run config written to" in joined
        assert "git_sha=" in joined
        assert "debug=" in joined

    def test_run_config_atomic_write(self):
        """Verify the .tmp file is moved into place (no partial writes left behind)."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp)
        orch._write_run_config()
        # After the write, no .tmp file should remain
        assert not (tmp / "run_config.json.tmp").exists()
        assert (tmp / "run_config.json").exists()

    def test_run_config_skipped_when_no_output_folder(self):
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
        orch.output_folder = None
        # Must not raise even though we never set the other attributes
        orch._write_run_config()

    def test_run_config_handles_non_json_serializable_params(self):
        """default=str fallback handles unusual CLI param values."""
        import tempfile, json as _json
        from pathlib import Path as _Path
        tmp = Path(tempfile.mkdtemp())
        orch = self._make_orchestrator(tmp, params={
            "action_space_file": "inputs/foo.csv",
            "some_path": _Path("/tmp/x"),  # not natively JSON serializable
        })
        orch._write_run_config()  # must NOT raise
        with open(tmp / "run_config.json") as f:
            data = _json.load(f)
        assert data["cli_params"]["some_path"] == "/tmp/x"


if __name__ == "__main__":
    unittest.main()
