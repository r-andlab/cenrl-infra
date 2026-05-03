"""Tests for state persistence primitives (Phase 02 Plan 01).

Covers:
- ActionSpaceBase.checkpoint_save() — mutation-safe GraphML write (Task 1)
- Infrastructure.utils.persistence.find_latest_state — state file detection (Task 2)
- Infrastructure.utils.persistence.load_graph_and_sidecar — graph + JSON load (Task 2)
- Infrastructure.utils.persistence.validate_sidecar — schema validation (Task 2)
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest

from models.base.action_space import (
    ActionSpaceBase,
    PARENTS,
    SLEEPING,
    EXPLORED,
    IS_TARGET_NODE,
    Q_VALUE,
    create_default_node_attributes,
    add_root,
    ROOT_KEY,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_minimal_action_space(tmp_path: Path) -> ActionSpaceBase:
    """Build a minimal ActionSpaceBase with a small handcrafted graph.

    Bypasses the heavy CSV-driven build_graph() path by mocking it out and
    constructing a tiny graph directly. Returns an instance whose self._graph
    has: root, two arm nodes, two target nodes (one sleeping with non-empty
    PARENTS list to exercise the orphan-reconnect path).
    """
    # Patch build_graph to skip CSV processing, and patch save() to no-op
    # (the constructor calls save() at the end, which would mutate the graph).
    g = nx.DiGraph()
    add_root(g)

    # Build two arms under root
    g.add_node(
        "arm_a",
        **create_default_node_attributes("arm_a", "arm_a", "category"),
    )
    g.add_edge(ROOT_KEY, "arm_a")
    g.add_node(
        "arm_b",
        **create_default_node_attributes("arm_b", "arm_b", "category"),
    )
    g.add_edge(ROOT_KEY, "arm_b")

    # Two target nodes under arm_a
    g.add_node(
        "target_1",
        **create_default_node_attributes(
            "target_1", "target_1", "domain", is_target_node=True
        ),
    )
    g.add_edge("arm_a", "target_1")

    g.add_node(
        "target_2",
        **create_default_node_attributes(
            "target_2", "target_2", "domain", is_target_node=True
        ),
    )
    g.add_edge("arm_a", "target_2")

    # An orphaned sleeping node: edge removed, parents list populated
    g.add_node(
        "sleeping_target",
        **create_default_node_attributes(
            "sleeping_target",
            "sleeping_target",
            "domain",
            is_target_node=True,
            sleeping=True,
        ),
    )
    g.add_edge("arm_b", "sleeping_target")
    # Simulate having been disconnected: remove edge, populate PARENTS
    g.remove_edge("arm_b", "sleeping_target")
    g.nodes["sleeping_target"][PARENTS] = ["arm_b"]

    df = None  # not used because build_graph is mocked

    with patch.object(ActionSpaceBase, "build_graph", return_value=g):
        with patch.object(ActionSpaceBase, "save", return_value=True):
            instance = ActionSpaceBase(
                output_directory=str(tmp_path),
                df=df,
                features=["category"],
                target_feature="domain",
            )

    # Sanity: ensure our handcrafted graph is the live graph
    assert instance._graph is g
    return instance


# -----------------------------------------------------------------------
# Task 1: ActionSpaceBase.checkpoint_save()
# -----------------------------------------------------------------------


class TestCheckpointSave:
    """Tests for ActionSpaceBase.checkpoint_save() — mutation-safe GraphML write."""

    def test_writes_valid_graphml_file(self, tmp_path):
        """checkpoint_save() writes a file readable by nx.read_graphml()."""
        space = _make_minimal_action_space(tmp_path)
        out_path = tmp_path / "checkpoint.graphml"

        result = space.checkpoint_save(str(out_path))

        assert result is True
        assert out_path.exists()
        # Round-trip: nx.read_graphml should parse it without error
        loaded = nx.read_graphml(str(out_path))
        assert loaded.number_of_nodes() == space._graph.number_of_nodes()

    def test_does_not_mutate_live_graph_node_count(self, tmp_path):
        """After checkpoint_save(), live graph node count is unchanged."""
        space = _make_minimal_action_space(tmp_path)
        before_nodes = space._graph.number_of_nodes()
        out_path = tmp_path / "checkpoint.graphml"

        space.checkpoint_save(str(out_path))

        assert space._graph.number_of_nodes() == before_nodes

    def test_does_not_mutate_live_graph_edge_count(self, tmp_path):
        """After checkpoint_save(), live graph edge count is unchanged.

        Specifically verifies that the orphan-reconnect path does not add
        edges to the live graph (only the deepcopy).
        """
        space = _make_minimal_action_space(tmp_path)
        before_edges = space._graph.number_of_edges()
        out_path = tmp_path / "checkpoint.graphml"

        space.checkpoint_save(str(out_path))

        # Live graph edge count must NOT have changed (no reconnect side effects)
        assert space._graph.number_of_edges() == before_edges

    def test_does_not_mutate_live_parents_attribute(self, tmp_path):
        """PARENTS values on live graph remain Python lists, not "" strings."""
        space = _make_minimal_action_space(tmp_path)
        # Before: sleeping_target has PARENTS = ["arm_b"]
        assert space._graph.nodes["sleeping_target"][PARENTS] == ["arm_b"]
        out_path = tmp_path / "checkpoint.graphml"

        space.checkpoint_save(str(out_path))

        # After: still a list, still contains "arm_b"
        live_parents = space._graph.nodes["sleeping_target"][PARENTS]
        assert isinstance(live_parents, list)
        assert live_parents == ["arm_b"]
        # Other nodes should still have empty list (not "")
        assert space._graph.nodes["target_1"][PARENTS] == []
        assert isinstance(space._graph.nodes["target_1"][PARENTS], list)

    def test_parents_serialized_as_empty_string_in_file(self, tmp_path):
        """PARENTS list values are converted to "" in the written file (no NetworkXError)."""
        space = _make_minimal_action_space(tmp_path)
        out_path = tmp_path / "checkpoint.graphml"

        # Should not raise — list-to-string conversion happens on the copy
        space.checkpoint_save(str(out_path))

        loaded = nx.read_graphml(str(out_path))
        # Every node in the loaded graph must have parents="" (not a list)
        for n, d in loaded.nodes(data=True):
            assert d.get("parents", "") == "", (
                f"node {n} has unexpected parents value: {d.get('parents')!r}"
            )

    def test_orphan_sleeping_nodes_reconnected_in_file(self, tmp_path):
        """Orphan sleeping nodes have their parent edges restored in the written file."""
        space = _make_minimal_action_space(tmp_path)
        # Pre-condition: sleeping_target has no incoming edge in the live graph
        assert ("arm_b", "sleeping_target") not in space._graph.edges
        out_path = tmp_path / "checkpoint.graphml"

        space.checkpoint_save(str(out_path))

        loaded = nx.read_graphml(str(out_path))
        # The written file should have the edge restored
        assert loaded.has_edge("arm_b", "sleeping_target"), (
            "orphan sleeping node was not reconnected in serialized GraphML"
        )

    def test_atomic_write_no_tmp_left_behind(self, tmp_path):
        """checkpoint_save() leaves no .tmp file behind on success."""
        space = _make_minimal_action_space(tmp_path)
        out_path = tmp_path / "checkpoint.graphml"

        space.checkpoint_save(str(out_path))

        assert out_path.exists()
        # The .tmp file used during atomic write must have been replaced
        assert not (tmp_path / "checkpoint.graphml.tmp").exists()

    def test_atomic_write_uses_os_replace(self, tmp_path):
        """checkpoint_save() uses os.replace for the final rename (atomic on POSIX).

        Verifies via mocking that os.replace is invoked with the .tmp path
        and the destination path.
        """
        space = _make_minimal_action_space(tmp_path)
        out_path = tmp_path / "checkpoint.graphml"

        with patch(
            "models.base.action_space.os.replace",
            wraps=os.replace,
        ) as mock_replace:
            space.checkpoint_save(str(out_path))

        # os.replace should have been called once with (tmp, final)
        assert mock_replace.call_count == 1
        call_args = mock_replace.call_args[0]
        assert call_args[0].endswith(".tmp")
        assert call_args[1] == str(out_path)


# -----------------------------------------------------------------------
# Task 2: persistence.find_latest_state
# -----------------------------------------------------------------------


class TestFindLatestState:
    """Tests for find_latest_state() — preference order and fallback."""

    def test_prefers_rolling_checkpoint(self, tmp_path):
        """Returns checkpoint files when both checkpoint.graphml and checkpoint.json exist."""
        from Infrastructure.utils.persistence import find_latest_state

        (tmp_path / "checkpoint.graphml").write_text("<graphml/>")
        (tmp_path / "checkpoint.json").write_text("{}")
        # Also create dated files; rolling checkpoint should win
        (tmp_path / "action_space_2026-04-24.graphml").write_text("<graphml/>")
        (tmp_path / "state_2026-04-24.json").write_text("{}")

        graphml, sidecar = find_latest_state(tmp_path)

        assert graphml == tmp_path / "checkpoint.graphml"
        assert sidecar == tmp_path / "checkpoint.json"

    def test_falls_back_to_most_recent_dated(self, tmp_path):
        """Falls back to most recent dated file when no checkpoint exists."""
        from Infrastructure.utils.persistence import find_latest_state

        (tmp_path / "action_space_2026-04-22.graphml").write_text("<graphml/>")
        (tmp_path / "state_2026-04-22.json").write_text("{}")
        (tmp_path / "action_space_2026-04-24.graphml").write_text("<graphml/>")
        (tmp_path / "state_2026-04-24.json").write_text("{}")
        (tmp_path / "action_space_2026-04-23.graphml").write_text("<graphml/>")
        (tmp_path / "state_2026-04-23.json").write_text("{}")

        graphml, sidecar = find_latest_state(tmp_path)

        # Lexicographic sort on ISO dates picks 2026-04-24 as most recent
        assert graphml == tmp_path / "action_space_2026-04-24.graphml"
        assert sidecar == tmp_path / "state_2026-04-24.json"

    def test_returns_none_when_empty_dir(self, tmp_path):
        """Returns (None, None) when state_dir is empty."""
        from Infrastructure.utils.persistence import find_latest_state

        graphml, sidecar = find_latest_state(tmp_path)

        assert graphml is None
        assert sidecar is None

    def test_returns_none_when_graphml_without_json(self, tmp_path):
        """Returns (None, None) when graphml exists but json is missing."""
        from Infrastructure.utils.persistence import find_latest_state

        # checkpoint.graphml without checkpoint.json
        (tmp_path / "checkpoint.graphml").write_text("<graphml/>")

        graphml, sidecar = find_latest_state(tmp_path)

        assert graphml is None
        assert sidecar is None

    def test_returns_none_when_dated_graphml_without_json(self, tmp_path):
        """Returns (None, None) when only dated graphml exists without matching json."""
        from Infrastructure.utils.persistence import find_latest_state

        (tmp_path / "action_space_2026-04-24.graphml").write_text("<graphml/>")
        # No matching state_2026-04-24.json

        graphml, sidecar = find_latest_state(tmp_path)

        assert graphml is None
        assert sidecar is None


# -----------------------------------------------------------------------
# Task 2: persistence.load_graph_and_sidecar
# -----------------------------------------------------------------------


def _write_minimal_graphml(path: Path, with_string_bools: bool = False) -> None:
    """Write a minimal GraphML containing one node with key attributes.

    If with_string_bools=True, encodes SLEEPING/EXPLORED/IS_TARGET_NODE as
    Python strings (legacy file format) to exercise the bool-cast guard.
    """
    g = nx.DiGraph()
    if with_string_bools:
        # Force string-typed booleans by storing them as Python strings
        g.add_node("n1", sleeping="True", explored="False", is_target_node="True")
    else:
        g.add_node("n1", sleeping=True, explored=False, is_target_node=True)
    # PARENTS as empty string (matches the on-disk format)
    g.nodes["n1"]["parents"] = ""
    nx.write_graphml(g, str(path))


def _valid_sidecar() -> dict:
    return {
        "exploration_epoch_num": 5,
        "current_epoch_num": 100,
        "c": 1.4,
        "step_size": 0.1,
        "initial_value_estimate": 0.0,
        "selection_method": "TOP_K_FROM_ARM",
        "size_method": "CONSTANT_VAL",
        "prop_method": "ON_RECEIPT",
        "save_timestamp": "2026-04-24T12:00:00+00:00",
        "country_code": "Germany",
    }


class TestLoadGraphAndSidecar:
    """Tests for load_graph_and_sidecar() — happy path, bool round-trip, corruption."""

    def test_loads_valid_graph_and_sidecar(self, tmp_path):
        """Loads a valid GraphML + JSON and returns (graph, sidecar)."""
        from Infrastructure.utils.persistence import load_graph_and_sidecar

        graphml_path = tmp_path / "checkpoint.graphml"
        json_path = tmp_path / "checkpoint.json"
        _write_minimal_graphml(graphml_path)
        with open(json_path, "w") as f:
            json.dump(_valid_sidecar(), f)

        graph, sidecar = load_graph_and_sidecar(graphml_path, json_path)

        assert graph is not None
        assert sidecar is not None
        assert graph.number_of_nodes() == 1
        assert sidecar["exploration_epoch_num"] == 5

    def test_casts_string_booleans_to_python_bool(self, tmp_path):
        """Casts string "True"/"False" to Python bool for SLEEPING, EXPLORED, IS_TARGET_NODE (D-11)."""
        from Infrastructure.utils.persistence import load_graph_and_sidecar

        graphml_path = tmp_path / "checkpoint.graphml"
        json_path = tmp_path / "checkpoint.json"
        _write_minimal_graphml(graphml_path, with_string_bools=True)
        with open(json_path, "w") as f:
            json.dump(_valid_sidecar(), f)

        graph, _ = load_graph_and_sidecar(graphml_path, json_path)

        node = graph.nodes["n1"]
        # All three bool attrs must be Python bool, not str
        assert node[SLEEPING] is True
        assert node[EXPLORED] is False
        assert node[IS_TARGET_NODE] is True
        assert isinstance(node[SLEEPING], bool)
        assert isinstance(node[EXPLORED], bool)
        assert isinstance(node[IS_TARGET_NODE], bool)

    def test_restores_parents_to_empty_list(self, tmp_path):
        """Restores PARENTS from "" to [] on all nodes."""
        from Infrastructure.utils.persistence import load_graph_and_sidecar

        graphml_path = tmp_path / "checkpoint.graphml"
        json_path = tmp_path / "checkpoint.json"
        _write_minimal_graphml(graphml_path)
        with open(json_path, "w") as f:
            json.dump(_valid_sidecar(), f)

        graph, _ = load_graph_and_sidecar(graphml_path, json_path)

        node = graph.nodes["n1"]
        assert node[PARENTS] == []
        assert isinstance(node[PARENTS], list)

    def test_returns_none_on_corrupted_graphml(self, tmp_path):
        """Returns (None, None) on corrupted/unreadable file (D-12)."""
        from Infrastructure.utils.persistence import load_graph_and_sidecar

        graphml_path = tmp_path / "checkpoint.graphml"
        json_path = tmp_path / "checkpoint.json"
        # Write garbage
        graphml_path.write_text("not valid xml at all")
        with open(json_path, "w") as f:
            json.dump(_valid_sidecar(), f)

        graph, sidecar = load_graph_and_sidecar(graphml_path, json_path)

        assert graph is None
        assert sidecar is None

    def test_returns_none_on_corrupted_json(self, tmp_path):
        """Returns (None, None) when JSON sidecar is malformed."""
        from Infrastructure.utils.persistence import load_graph_and_sidecar

        graphml_path = tmp_path / "checkpoint.graphml"
        json_path = tmp_path / "checkpoint.json"
        _write_minimal_graphml(graphml_path)
        json_path.write_text("{ this is not valid json")

        graph, sidecar = load_graph_and_sidecar(graphml_path, json_path)

        assert graph is None
        assert sidecar is None


# -----------------------------------------------------------------------
# Task 2: persistence.validate_sidecar
# -----------------------------------------------------------------------


class TestValidateSidecar:
    """Tests for validate_sidecar() — required-key schema check."""

    def test_returns_true_when_all_keys_present(self):
        """Returns True when all required keys present."""
        from Infrastructure.utils.persistence import validate_sidecar

        assert validate_sidecar(_valid_sidecar()) is True

    def test_returns_false_when_keys_missing(self):
        """Returns False when keys are missing."""
        from Infrastructure.utils.persistence import validate_sidecar

        sidecar = _valid_sidecar()
        del sidecar["c"]
        del sidecar["exploration_epoch_num"]

        assert validate_sidecar(sidecar) is False

    def test_returns_false_for_empty_dict(self):
        """Returns False for empty dict."""
        from Infrastructure.utils.persistence import validate_sidecar

        assert validate_sidecar({}) is False
