import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from Infrastructure.utils.structures import NodeState


class TestSoftReset(unittest.TestCase):
    """Tests for RegionalNode.soft_reset() -- covers RUNT-03."""

    def _make_node(self):
        from Infrastructure.main.node import RegionalNode
        with patch.object(RegionalNode, '__init__', lambda self, *a, **kw: None):
            node = RegionalNode.__new__(RegionalNode)
        node.country = "TestCountry"
        node.model = MagicMock()
        node.model.action_space = MagicMock()
        node.model.action_space.wake_up_all_nodes = MagicMock()
        node.model.current_epoch_num = 50
        node.state = NodeState.AWAITING_MEASUREMENTS
        node.in_flight = 3
        node.episode_stats = [{"episode": 1, "time": 1}]
        return node

    def test_soft_reset_preserves_qvalues(self):
        """soft_reset() must NOT call model.reset() -- Q-values are preserved."""
        node = self._make_node()
        node.soft_reset()
        node.model.action_space.wake_up_all_nodes.assert_called_once()
        node.model.reset.assert_not_called()

    def test_soft_reset_resets_epoch(self):
        """soft_reset() resets current_epoch_num to 0."""
        node = self._make_node()
        node.model.current_epoch_num = 75
        node.soft_reset()
        self.assertEqual(node.model.current_epoch_num, 0)

    def test_soft_reset_clears_inflight(self):
        """soft_reset() sets in_flight to 0."""
        node = self._make_node()
        node.in_flight = 5
        node.soft_reset()
        self.assertEqual(node.in_flight, 0)

    def test_soft_reset_sets_idle(self):
        """soft_reset() sets state to IDLE."""
        node = self._make_node()
        node.soft_reset()
        self.assertEqual(node.state, NodeState.IDLE)


class TestFinishEpisodeIdleNotDone(unittest.TestCase):
    """Tests for _finish_episode_if_ready() -- covers D-06/D-07."""

    def _make_node(self):
        from Infrastructure.main.node import RegionalNode
        with patch.object(RegionalNode, '__init__', lambda self, *a, **kw: None):
            node = RegionalNode.__new__(RegionalNode)
        node.country = "TestCountry"
        node.model = MagicMock()
        node.model.measurements_per_episode = 100
        node.model.current_epoch_num = 100  # quota reached
        node.model.num_episodes = 1
        node.state = NodeState.READY
        node.in_flight = 0
        node.episode_stats = [{"episode": 1, "time": 1}]
        node.episode_all_stats = []
        node.episode_idx = 1
        node.stat_df = None
        node.save_stats = True
        return node

    def test_finish_episode_sets_idle_not_done(self):
        """When quota is reached, state becomes IDLE not DONE."""
        node = self._make_node()
        node._finish_episode_if_ready(save_stats=True)
        self.assertEqual(node.state, NodeState.IDLE)
        self.assertNotEqual(node.state, NodeState.DONE)

    def test_finish_episode_does_not_call_reset(self):
        """_finish_episode_if_ready must NOT call model.reset()."""
        node = self._make_node()
        node._finish_episode_if_ready(save_stats=True)
        node.model.reset.assert_not_called()

    def test_finish_episode_noop_when_inflight(self):
        """Does nothing when in_flight > 0."""
        node = self._make_node()
        node.in_flight = 3
        result = node._finish_episode_if_ready(save_stats=True)
        self.assertIsNone(result)
        self.assertEqual(node.state, NodeState.READY)  # unchanged


class TestWriteStatsNullSafe(unittest.TestCase):
    """Tests for write_stats() null-safety -- covers SIGTERM early shutdown."""

    def _make_node(self):
        from Infrastructure.main.node import RegionalNode
        with patch.object(RegionalNode, '__init__', lambda self, *a, **kw: None):
            node = RegionalNode.__new__(RegionalNode)
        node.country = "TestCountry"
        node.model = MagicMock()
        node.model.outfile = "/tmp/test_stats.csv"
        node.model.save = MagicMock()
        node.stat_df = None
        node.episode_stats = []
        node.episode_all_stats = []
        return node

    def test_write_stats_no_crash_when_stat_df_none(self):
        """write_stats() must not raise when stat_df is None and no stats exist."""
        node = self._make_node()
        # Should not raise
        node.write_stats()
        node.model.save.assert_called_once()

    def test_write_stats_flushes_partial_stats(self):
        """write_stats() flushes episode_all_stats when stat_df is None."""
        node = self._make_node()
        node.episode_all_stats = [{"episode": 1, "time": 1, "reward": 0.5}]
        with patch.object(pd.DataFrame, 'to_csv') as mock_csv:
            node.write_stats()
            mock_csv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
