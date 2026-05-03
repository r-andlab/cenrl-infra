import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from Infrastructure.utils.structures import (
    NodeState,
    BatchSelectionMethod,
    BatchSizeMethod,
    PropagationMethod,
)


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


class TestSaveCheckpoint(unittest.TestCase):
    """Tests for RegionalNode.save_checkpoint() / save_daily() / _write_sidecar()."""

    def _make_node(self, tmp_path):
        from Infrastructure.main.node import RegionalNode
        with patch.object(RegionalNode, '__init__', lambda self, *a, **kw: None):
            node = RegionalNode.__new__(RegionalNode)
        node.country = "TestCountry"
        node.country_name_standard = "TestCountry"
        node.model = MagicMock()
        node.model.action_space = MagicMock()
        node.model.action_space.checkpoint_save = MagicMock(return_value=True)
        node.model.exploration_epoch_num = 42
        node.model.current_epoch_num = 17
        node.model.c = 1.5
        node.model.step_size = 0.1
        node.model.initial_value_estimate = 1.0
        node.model.selection_method = BatchSelectionMethod.TOP_K_FROM_ARM
        node.model.size_method = BatchSizeMethod.CONSTANT_VAL
        node.model.prop_method = PropagationMethod.ON_RECEIPT
        node.model.output_directory = str(tmp_path)
        return node

    def test_save_checkpoint_creates_graphml_and_json(self):
        """save_checkpoint() creates checkpoint.graphml and checkpoint.json in state_dir."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            state_dir = tmp_path / "state"

            node.save_checkpoint(state_dir)

            # action_space.checkpoint_save should be called with the graphml path
            node.model.action_space.checkpoint_save.assert_called_once()
            called_path = node.model.action_space.checkpoint_save.call_args[0][0]
            self.assertTrue(called_path.endswith("checkpoint.graphml"))
            # JSON sidecar should be written
            self.assertTrue((state_dir / "checkpoint.json").exists())

    def test_save_checkpoint_overwrites_existing_files(self):
        """save_checkpoint() overwrites existing checkpoint files (rolling, per D-02)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            state_dir = tmp_path / "state"
            state_dir.mkdir(parents=True)
            # Pre-populate the checkpoint.json so we can confirm overwrite
            (state_dir / "checkpoint.json").write_text('{"old": "content"}')

            node.save_checkpoint(state_dir)

            content = (state_dir / "checkpoint.json").read_text()
            data = json.loads(content)
            # Old content should be gone; new sidecar fields should be present
            self.assertNotIn("old", data)
            self.assertEqual(data["exploration_epoch_num"], 42)

    def test_save_daily_uses_date_stamped_filenames(self):
        """save_daily('2026-04-24') creates action_space_2026-04-24.graphml and state_2026-04-24.json."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            state_dir = tmp_path / "state"

            node.save_daily("2026-04-24", state_dir)

            called_path = node.model.action_space.checkpoint_save.call_args[0][0]
            self.assertTrue(called_path.endswith("action_space_2026-04-24.graphml"))
            self.assertTrue((state_dir / "state_2026-04-24.json").exists())

    def test_write_sidecar_contains_all_required_keys(self):
        """_write_sidecar() writes JSON with all REQUIRED_SIDECAR_KEYS plus metadata."""
        import tempfile
        from Infrastructure.utils.persistence import REQUIRED_SIDECAR_KEYS
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            json_path = tmp_path / "test.json"

            node._write_sidecar(json_path)

            self.assertTrue(json_path.exists())
            data = json.loads(json_path.read_text())
            for key in REQUIRED_SIDECAR_KEYS:
                self.assertIn(key, data, f"Missing required key: {key}")
            # Metadata fields per D-14
            self.assertIn("save_timestamp", data)
            self.assertIn("country_code", data)
            self.assertIn("measurements_completed_today", data)
            # Verify scalar values are correctly captured
            self.assertEqual(data["exploration_epoch_num"], 42)
            self.assertEqual(data["current_epoch_num"], 17)
            self.assertEqual(data["c"], 1.5)
            self.assertEqual(data["country_code"], "TestCountry")

    def test_write_sidecar_serializes_enums_as_name_strings(self):
        """_write_sidecar() serializes enums as .name strings (e.g., "TOP_K_FROM_ARM")."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            json_path = tmp_path / "test.json"

            node._write_sidecar(json_path)

            data = json.loads(json_path.read_text())
            self.assertEqual(data["selection_method"], "TOP_K_FROM_ARM")
            self.assertEqual(data["size_method"], "CONSTANT_VAL")
            self.assertEqual(data["prop_method"], "ON_RECEIPT")

    def test_write_sidecar_uses_atomic_write(self):
        """_write_sidecar() uses atomic write (.tmp + os.replace)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            json_path = tmp_path / "test.json"

            with patch("Infrastructure.main.node.os.replace", wraps=os.replace) as mock_replace:
                node._write_sidecar(json_path)

            mock_replace.assert_called_once()
            args = mock_replace.call_args[0]
            self.assertTrue(args[0].endswith(".tmp"))
            self.assertEqual(args[1], str(json_path))
            # No leftover .tmp file
            self.assertFalse((tmp_path / "test.json.tmp").exists())

    def test_save_checkpoint_logs_error_on_failure_no_crash(self):
        """save_checkpoint() catches exceptions and logs error without crashing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            # Make checkpoint_save raise
            node.model.action_space.checkpoint_save = MagicMock(
                side_effect=RuntimeError("disk full")
            )
            state_dir = tmp_path / "state"

            # Should not raise
            try:
                node.save_checkpoint(state_dir)
            except Exception as exc:
                self.fail(f"save_checkpoint should swallow exceptions, raised: {exc}")


if __name__ == "__main__":
    unittest.main()
