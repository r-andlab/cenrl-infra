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


class TestLoadCheckpoint(unittest.TestCase):
    """Tests for RegionalNode.load_checkpoint() — covers D-09, D-10, D-11, D-12."""

    def _make_node(self, tmp_path):
        """Build a node with a mocked model whose action_space has a real graph."""
        import networkx as nx
        from models.base.action_space import (
            create_default_node_attributes,
            add_root,
            ROOT_KEY,
            Q_VALUE,
            ACTION_ATTEMPTS,
            SLEEPING,
            EXPLORED,
            IS_TARGET_NODE,
            PARENTS,
        )

        from Infrastructure.main.node import RegionalNode
        with patch.object(RegionalNode, '__init__', lambda self, *a, **kw: None):
            node = RegionalNode.__new__(RegionalNode)
        node.country = "TestCountry"
        node.country_name_standard = "TestCountry"
        node.model = MagicMock()
        node.model.exploration_epoch_num = 0
        node.model.current_epoch_num = 0
        node.model.c = 1.0
        node.model.step_size = None
        node.model.initial_value_estimate = 1.0
        node.model.selection_method = BatchSelectionMethod.TOP_K_FROM_ARM
        node.model.size_method = BatchSizeMethod.CONSTANT_VAL
        node.model.prop_method = PropagationMethod.ON_RECEIPT
        node.model.output_directory = str(tmp_path)

        # Build a minimal "fresh" current graph: root, two arms, three targets.
        g = nx.DiGraph()
        add_root(g)
        g.add_node(
            "arm_a",
            **create_default_node_attributes("arm_a", "arm_a", "category"),
        )
        g.add_edge(ROOT_KEY, "arm_a")
        for t in ("target_1", "target_2", "target_3"):
            g.add_node(
                t,
                **create_default_node_attributes(t, t, "domain", is_target_node=True),
            )
            g.add_edge("arm_a", t)
        node.model.action_space = MagicMock()
        node.model.action_space._graph = g
        return node

    def _write_saved_state(self, state_dir: Path, node_data_overrides=None,
                           sidecar_overrides=None):
        """Helper: write a synthetic checkpoint.graphml + checkpoint.json pair."""
        import networkx as nx
        from models.base.action_space import (
            create_default_node_attributes,
            add_root,
            ROOT_KEY,
            Q_VALUE,
            ACTION_ATTEMPTS,
        )

        state_dir.mkdir(parents=True, exist_ok=True)

        saved = nx.DiGraph()
        add_root(saved)
        saved.add_node(
            "arm_a",
            **create_default_node_attributes("arm_a", "arm_a", "category"),
        )
        saved.add_edge(ROOT_KEY, "arm_a")
        # target_1 has learned Q-value; target_2 also; target_REMOVED not in current CSV
        for t, q in (("target_1", 0.75), ("target_2", 0.42), ("target_REMOVED", 0.99)):
            attrs = create_default_node_attributes(t, t, "domain", is_target_node=True)
            attrs[Q_VALUE] = q
            attrs[ACTION_ATTEMPTS] = 5
            saved.add_node(t, **attrs)
            saved.add_edge("arm_a", t)
        # GraphML cannot serialize lists; convert PARENTS to ""
        for n, d in saved.nodes(data=True):
            d["parents"] = ""

        graphml_path = state_dir / "checkpoint.graphml"
        nx.write_graphml(saved, str(graphml_path))

        sidecar = {
            "exploration_epoch_num": 100,
            "current_epoch_num": 50,
            "measurements_completed_today": 50,
            "c": 1.0,
            "step_size": None,
            "initial_value_estimate": 1.0,
            "selection_method": "TOP_K_FROM_ARM",
            "size_method": "CONSTANT_VAL",
            "prop_method": "ON_RECEIPT",
            "save_timestamp": "2026-04-24T12:00:00+00:00",
            "country_code": "TestCountry",
        }
        if sidecar_overrides:
            sidecar.update(sidecar_overrides)
        json_path = state_dir / "checkpoint.json"
        json_path.write_text(json.dumps(sidecar))
        return graphml_path, json_path

    def test_load_checkpoint_returns_true_and_restores_scalars(self):
        """load_checkpoint() returns True for valid state and restores scalars."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            state_dir = tmp_path / "state"
            self._write_saved_state(state_dir)

            result = node.load_checkpoint(state_dir)

            self.assertTrue(result)
            self.assertEqual(node.model.exploration_epoch_num, 100)
            self.assertEqual(node.model.current_epoch_num, 50)

    def test_load_checkpoint_returns_false_when_no_state(self):
        """load_checkpoint() returns False when no state files exist."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            state_dir = tmp_path / "state"
            state_dir.mkdir(parents=True)

            result = node.load_checkpoint(state_dir)

            self.assertFalse(result)
            # Scalars unchanged
            self.assertEqual(node.model.exploration_epoch_num, 0)

    def test_load_checkpoint_returns_false_on_corrupt_files(self):
        """load_checkpoint() returns False on corrupted state files (D-12)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            state_dir = tmp_path / "state"
            state_dir.mkdir(parents=True)
            # Write garbage to checkpoint files
            (state_dir / "checkpoint.graphml").write_text("not valid xml")
            (state_dir / "checkpoint.json").write_text("not valid json")

            result = node.load_checkpoint(state_dir)

            self.assertFalse(result)

    def test_load_checkpoint_graph_merge_preserves_qvalues(self):
        """D-10: Graph merge copies Q-values from saved graph to current graph
        for nodes present in both; new targets keep defaults; removed targets dropped."""
        import tempfile
        from models.base.action_space import Q_VALUE, ACTION_ATTEMPTS
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            state_dir = tmp_path / "state"
            self._write_saved_state(state_dir)

            result = node.load_checkpoint(state_dir)

            self.assertTrue(result)
            # target_1 and target_2 should have Q-values from saved state
            self.assertEqual(node.model.action_space._graph.nodes["target_1"][Q_VALUE], 0.75)
            self.assertEqual(node.model.action_space._graph.nodes["target_2"][Q_VALUE], 0.42)
            self.assertEqual(node.model.action_space._graph.nodes["target_1"][ACTION_ATTEMPTS], 5)
            # target_3 is in CSV but was not in saved state — keeps default 0
            self.assertEqual(node.model.action_space._graph.nodes["target_3"][Q_VALUE], 0)
            # target_REMOVED was in saved state but not in current CSV — must NOT appear
            self.assertFalse(node.model.action_space._graph.has_node("target_REMOVED"))

    def test_load_checkpoint_warns_on_hyperparameter_mismatch(self):
        """Hyperparameter mismatch logs a warning but still loads."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            # Saved state has c=1.0, current model has c=2.5
            node.model.c = 2.5
            state_dir = tmp_path / "state"
            self._write_saved_state(state_dir)

            with self.assertLogs("Infrastructure.main.node", level="WARNING") as cm:
                result = node.load_checkpoint(state_dir)

            self.assertTrue(result)
            # At least one warning mentioning hyperparameter mismatch
            self.assertTrue(
                any("mismatch" in msg.lower() or "c=" in msg for msg in cm.output),
                f"Expected hyperparameter-mismatch warning, got: {cm.output}",
            )

    def test_load_checkpoint_restores_enum_fields(self):
        """Enum fields (selection_method etc.) are restored from sidecar names."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            # Pre-set to different enum values to confirm restoration
            node.model.selection_method = BatchSelectionMethod.UNIFORM_SPREAD
            node.model.size_method = BatchSizeMethod.VARY_ON_SUCCESS
            node.model.prop_method = PropagationMethod.IN_ORDER
            state_dir = tmp_path / "state"
            self._write_saved_state(state_dir)

            result = node.load_checkpoint(state_dir)

            self.assertTrue(result)
            self.assertEqual(node.model.selection_method, BatchSelectionMethod.TOP_K_FROM_ARM)
            self.assertEqual(node.model.size_method, BatchSizeMethod.CONSTANT_VAL)
            self.assertEqual(node.model.prop_method, PropagationMethod.ON_RECEIPT)


class TestWriteStatsRollingAndSnapshot(unittest.TestCase):
    """Tests for write_stats() rolling-append + per-iteration snapshot (D-05, D-07)."""

    def _make_node(self, tmp_path: Path):
        from Infrastructure.main.node import RegionalNode
        with patch.object(RegionalNode, '__init__', lambda self, *a, **kw: None):
            node = RegionalNode.__new__(RegionalNode)
        node.country = "TestCountry"
        node.model = MagicMock()
        node.model.save = MagicMock()
        node.model.outfile = str(tmp_path / "TestCountry.csv")
        node.episode_idx = 1
        node.episode_stats = []
        node.episode_all_stats = []
        node.stat_df = None
        return node

    def test_first_call_writes_snapshot_and_creates_rolling_with_header(self):
        """First write_stats call: snapshot file created AND rolling CSV created with header."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            # Caller has rotated rows into episode_all_stats (Task 3 invariant).
            node.episode_all_stats = [
                {"episode": 1, "time": 1, "reward": 0.5},
                {"episode": 1, "time": 2, "reward": 0.7},
            ]

            node.write_stats()

            rolling_path = tmp_path / "TestCountry.csv"
            snapshot_path = tmp_path / "TestCountry_iter_001.csv"
            self.assertTrue(rolling_path.exists(), "rolling CSV not created")
            self.assertTrue(snapshot_path.exists(), "snapshot CSV not created")

            rolling_df = pd.read_csv(rolling_path)
            snapshot_df = pd.read_csv(snapshot_path)
            self.assertEqual(len(rolling_df), 2)
            self.assertEqual(len(snapshot_df), 2)
            # Header present in rolling on first write
            with open(rolling_path) as f:
                first_line = f.readline()
            self.assertIn("episode", first_line)

    def test_second_call_appends_to_rolling_without_header(self):
        """Second iteration: rolling CSV appended (no header rewritten); snapshot for iter_002 separate."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)

            # First iteration write
            node.episode_idx = 1
            node.episode_all_stats = [{"episode": 1, "time": 1, "reward": 0.5}]
            node.write_stats()

            # Second iteration write (caller bumped episode_idx and re-rotated rows)
            node.episode_idx = 2
            node.episode_all_stats = [
                {"episode": 2, "time": 1, "reward": 0.6},
                {"episode": 2, "time": 2, "reward": 0.8},
            ]
            node.write_stats()

            rolling_path = tmp_path / "TestCountry.csv"
            with open(rolling_path) as f:
                lines = f.readlines()

            # 1 header + 1 row from iter_001 + 2 rows from iter_002 = 4 lines
            self.assertEqual(len(lines), 4)
            # Header appears exactly once (first line). Subsequent lines must NOT contain "episode,time"
            header_count = sum(
                1 for line in lines if line.startswith("episode,")
            )
            self.assertEqual(header_count, 1, f"Expected 1 header line, got {header_count}")

            # iter_002 snapshot exists and has only iter_002 rows
            snapshot2 = tmp_path / "TestCountry_iter_002.csv"
            self.assertTrue(snapshot2.exists())
            snap_df = pd.read_csv(snapshot2)
            self.assertEqual(len(snap_df), 2)

    def test_per_iteration_snapshot_is_overwritten_not_appended(self):
        """Calling write_stats twice with the same episode_idx overwrites the snapshot (no append)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            node.episode_idx = 3

            node.episode_all_stats = [{"episode": 3, "time": 1, "reward": 0.1}]
            node.write_stats()

            # Re-set rows (write_stats clears episode_all_stats post-write)
            node.episode_all_stats = [{"episode": 3, "time": 1, "reward": 0.1}]
            node.write_stats()

            snapshot_path = tmp_path / "TestCountry_iter_003.csv"
            snap_df = pd.read_csv(snapshot_path)
            # Snapshot has only one row (overwrite, not append)
            self.assertEqual(len(snap_df), 1)

    def test_writes_use_episode_idx_in_snapshot_filename(self):
        """episode_idx=7 → snapshot path ends with _iter_007.csv (zero-padded)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            node.episode_idx = 7
            node.episode_all_stats = [{"episode": 7, "time": 1, "reward": 0.0}]

            node.write_stats()

            snapshot_path = tmp_path / "TestCountry_iter_007.csv"
            self.assertTrue(snapshot_path.exists(), f"Expected zero-padded snapshot at {snapshot_path}")

    def test_model_save_called_once_per_invocation(self):
        """write_stats must call self.model.save() exactly once per call."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            node.episode_all_stats = [{"episode": 1, "time": 1, "reward": 0.5}]

            node.write_stats()

            node.model.save.assert_called_once()

    def test_no_rows_no_files_written(self):
        """Empty episode_stats AND episode_all_stats: neither snapshot nor rolling CSV created."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            # Both buffers empty
            node.episode_stats = []
            node.episode_all_stats = []

            node.write_stats()

            rolling_path = tmp_path / "TestCountry.csv"
            snapshot_path = tmp_path / "TestCountry_iter_001.csv"
            self.assertFalse(rolling_path.exists(), "rolling CSV should not be created when no rows")
            self.assertFalse(snapshot_path.exists(), "snapshot CSV should not be created when no rows")
            # model.save still ran (writes action_space.csv side-table)
            node.model.save.assert_called_once()

    def test_clears_episode_all_stats_after_write(self):
        """Post-condition: episode_all_stats == [] after write_stats returns."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            node = self._make_node(tmp_path)
            node.episode_all_stats = [{"episode": 1, "time": 1, "reward": 0.5}]

            node.write_stats()

            self.assertEqual(node.episode_all_stats, [])


if __name__ == "__main__":
    unittest.main()
