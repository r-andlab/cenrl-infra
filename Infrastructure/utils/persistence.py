"""State persistence utilities for the Infrastructure orchestrator.

Provides three primitives consumed by RegionalNode (Plan 02) and Orchestrator
(Plan 03):

- find_latest_state: scan a per-country state directory and return the most
  recent (graphml, json) pair, preferring the rolling checkpoint over dated
  daily snapshots (D-09, D-06).
- load_graph_and_sidecar: read a GraphML + JSON pair, applying the D-11
  string-to-bool round-trip fix and restoring PARENTS from "" to []. On any
  error, returns (None, None) so callers can start fresh per D-12.
- validate_sidecar: check that a loaded sidecar contains the keys required
  to restore UCB scalar state (mitigates T-02-01 tampering).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# Node attribute names that must round-trip as Python bool (D-11). Stored
# as lower-case strings here because that is how they appear in the GraphML
# file produced by nx.write_graphml(); see action_space.py constants
# SLEEPING / EXPLORED / IS_TARGET_NODE.
BOOL_ATTRS = {"sleeping", "explored", "is_target_node"}

# Required keys for a valid UCB-state sidecar. Mirrors the fields written by
# RegionalNode._write_sidecar() (Plan 02). Missing any of these means we
# cannot safely restore model state, so we treat the sidecar as invalid.
REQUIRED_SIDECAR_KEYS = {
    "exploration_epoch_num",
    "current_epoch_num",
    "c",
    "step_size",
    "initial_value_estimate",
    "selection_method",
    "size_method",
    "prop_method",
}


def find_latest_state(state_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (graphml_path, json_path) for the most recent saved state.

    Preference order (D-09, D-06):
      1. Rolling checkpoint: ``checkpoint.graphml`` + ``checkpoint.json``
      2. Most recent dated daily save: ``action_space_<date>.graphml`` +
         ``state_<date>.json``

    Returns (None, None) when no usable pair exists. A graphml without its
    matching json sidecar is treated as missing — both files are required to
    restore UCB state.
    """
    ckpt_graphml = state_dir / "checkpoint.graphml"
    ckpt_json = state_dir / "checkpoint.json"
    if ckpt_graphml.exists() and ckpt_json.exists():
        return ckpt_graphml, ckpt_json

    # Fall back to the most recent dated daily save. ISO-8601 dates sort
    # lexicographically so reverse-sorted glob picks the newest first.
    dated = sorted(state_dir.glob("action_space_*.graphml"), reverse=True)
    for graphml_path in dated:
        date_str = graphml_path.stem.replace("action_space_", "")
        json_path = state_dir / f"state_{date_str}.json"
        if json_path.exists():
            return graphml_path, json_path

    return None, None


def load_graph_and_sidecar(
    graphml_path: Path, json_path: Path
) -> Tuple[Optional[Any], Optional[Dict]]:
    """Load a GraphML + JSON sidecar pair.

    Returns (graph, sidecar) on success, or (None, None) on any read or
    parse failure (D-12: corrupt files must not crash the orchestrator).

    On load, applies the D-11 boolean round-trip fix: any node attribute in
    BOOL_ATTRS that is currently a string ("True"/"False") is cast back to
    Python bool. NetworkX 3.5 round-trips Python bool correctly when the
    file was written with bool-typed values, but legacy files that stored
    booleans as strings produce string attribute values on load, which
    silently corrupt UCB sleeping-node logic if not corrected here.

    Also restores PARENTS from the empty-string GraphML representation back
    to an empty list, matching the in-memory invariant maintained by
    ActionSpaceBase.
    """
    try:
        g = nx.read_graphml(str(graphml_path))
        for _, d in g.nodes(data=True):
            for attr in BOOL_ATTRS:
                if attr in d and isinstance(d[attr], str):
                    d[attr] = d[attr].lower() == "true"
            # PARENTS is serialized as "" by checkpoint_save(); restore the
            # in-memory list invariant for downstream graph operations.
            d["parents"] = []

        with open(json_path) as f:
            sidecar = json.load(f)

        logger.info(
            "Loaded state from %s: %d nodes",
            graphml_path,
            g.number_of_nodes(),
        )
        return g, sidecar
    except Exception as exc:
        logger.error(
            "Failed to load state from %s: %s -- starting fresh",
            graphml_path,
            exc,
        )
        return None, None


def validate_sidecar(sidecar: Dict) -> bool:
    """Return True iff the sidecar contains every key in REQUIRED_SIDECAR_KEYS.

    Mitigates T-02-01: a tampered or truncated sidecar that is missing UCB
    scalar fields would silently restore the model with default values
    (e.g. exploration_epoch_num=0), corrupting learning. Callers that hit
    False should treat the state as unrecoverable and start fresh per D-12.
    """
    missing = REQUIRED_SIDECAR_KEYS - sidecar.keys()
    if missing:
        logger.error("Sidecar missing required keys: %s", sorted(missing))
        return False
    return True
