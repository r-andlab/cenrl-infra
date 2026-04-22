import logging
import signal
import time
import unittest
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
        orch._last_delay_warn = time.monotonic()
        orch.api = MagicMock()
        orch.api.receiver._process = None  # debug mode
        orch.target_countries = []
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
        orch._last_delay_warn = time.monotonic()
        orch.api = MagicMock()
        orch.api.receiver._process = None
        orch.target_countries = []
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
        orch._last_delay_warn = time.monotonic()
        orch.api = MagicMock()
        orch.api.receiver._process = None

        node = MagicMock()
        node.write_stats = MagicMock()
        node.in_flight = 0
        orch.agents = {"US": node}
        orch.target_countries = []

        # Set stop immediately
        orch._stop_event.set()
        orch._install_signal_handlers = MagicMock()

        orch.run_forever()
        node.write_stats.assert_called_once()


class TestDelayedResultLogging(unittest.TestCase):
    """Verify 5-min delayed-result logger -- covers RUNT-05/D-09."""

    def test_delayed_result_logging(self):
        """Warning logged when 5+ minutes elapsed and nodes have in-flight."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        node = MagicMock()
        node.in_flight = 3
        orch.agents = {"US": node}
        # Set last warn time to 6 minutes ago
        orch._last_delay_warn = time.monotonic() - 360

        with self.assertLogs(level=logging.WARNING) as cm:
            orch._check_delayed_results()
        self.assertTrue(any("DELAYED" in msg for msg in cm.output))
        self.assertTrue(any("US" in msg for msg in cm.output))

    def test_no_logging_before_5_minutes(self):
        """No warning when less than 5 minutes have passed."""
        from Infrastructure.main.orchestrator import Orchestrator
        with patch.object(Orchestrator, '__init__', lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)

        node = MagicMock()
        node.in_flight = 3
        orch.agents = {"US": node}
        orch._last_delay_warn = time.monotonic()  # just now

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


if __name__ == "__main__":
    unittest.main()
