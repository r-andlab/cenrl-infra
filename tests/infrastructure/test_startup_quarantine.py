import time
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from Infrastructure.main.orchestrator import Orchestrator


class _Payload:
    def __init__(self, vp, issue=None, template=None):
        self.vp = vp
        self.issue = issue
        self.template = template


def _make(all_vps=True):
    orch = Orchestrator.__new__(Orchestrator)
    orch._all_vps_mode = all_vps
    orch.eval_store = MagicMock()
    orch.eval_store.get_country.return_value = "Germany"
    orch.vantage_points = MagicMock()
    orch.api = MagicMock()
    orch.api.debug = False
    orch.services = ["https"]
    orch.agents = {}
    orch.completed_agents = set()
    orch.target_countries = {"Germany"}
    orch.params = {}
    orch._vp_failure_counts = defaultdict(int)
    orch.output_folder = None
    orch.previous_values_folder = None
    orch._model_klass = MagicMock()
    orch._target_selection = MagicMock()
    orch._batch_size_method = MagicMock()
    orch._qval_propagation_method = MagicMock()
    orch._aggregation_method = MagicMock()
    orch._reward_mode = "percentage"
    orch._on_country_reset = MagicMock()
    return orch


class TestStartupFailureQuarantine(unittest.TestCase):
    def test_failed_startup_vp_is_quarantined_not_removed(self):
        orch = _make(all_vps=True)
        orch.eval_store.drain.return_value = [_Payload("1.2.3.4", issue="reset")]
        orch._process_eval_results()
        orch.vantage_points.reject_vp.assert_not_called()
        orch.api.remove_vantage_points.assert_not_called()
        self.assertIn(("Germany", "1.2.3.4"), orch._quarantine_map())
        self.assertIn(("Germany", "1.2.3.4"), orch._startup_retest())

    def test_startup_failure_stays_out_of_dispatch_after_release(self):
        orch = _make(all_vps=True)
        orch.eval_store.drain.return_value = [_Payload("1.2.3.4", issue="reset")]
        orch._process_eval_results()
        orch._quarantine_map()[("Germany", "1.2.3.4")] = time.monotonic() - 1
        orch._release_expired_quarantine()
        self.assertEqual(orch._available_vps("Germany", ["1.2.3.4", "5.6.7.8"]), ["5.6.7.8"])

    def test_passing_retest_clears_startup_state(self):
        orch = _make(all_vps=True)
        orch._startup_retest().add(("Germany", "1.2.3.4"))
        orch.eval_store.drain.return_value = [_Payload("1.2.3.4", issue=None, template={"x": 1})]
        with patch("Infrastructure.main.orchestrator.RegionalNode"):
            orch._process_eval_results()
        self.assertNotIn(("Germany", "1.2.3.4"), orch._startup_retest())

    def test_non_all_vps_mode_still_rejects_failed_vp(self):
        orch = _make(all_vps=False)
        orch.vantage_points.reject_vp.return_value = None
        orch.eval_store.drain.return_value = [_Payload("1.2.3.4", issue="reset")]
        orch._process_eval_results()
        orch.vantage_points.reject_vp.assert_called_once_with("Germany", "1.2.3.4")
        orch.api.remove_vantage_points.assert_called_once()


if __name__ == "__main__":
    unittest.main()
