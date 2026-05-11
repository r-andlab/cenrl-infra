"""Shared style + IO helpers for the Infrastructure plotting CLIs.

Module-level functions only. Imported by the three per-stream scripts
(plot_tick_timings.py, plot_measurements.py, plot_accuracy.py) to keep
the figure-emission code free of plumbing.

Matches the repo's existing paper-plot style (see scripts/plot_utils.py):
seaborn paper context, ticks style, pdf.fonttype=42 for matplotlib PDF
embedding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # noqa: F401  (re-exported for downstream convenience)
import pandas as pd  # noqa: F401  (re-exported for downstream convenience)
import seaborn as sns

logger = logging.getLogger(__name__)


def apply_paper_style() -> None:
    """Set seaborn paper context, ticks style, and pdf.fonttype=42.

    Matches scripts/plot_utils.py lines 60-62 so the new Infrastructure
    plots have the same visual identity as the controlled-eval plots.
    """
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams["pdf.fonttype"] = 42


def load_run_config(run_dir: Path) -> dict:
    """Return run_config.json contents, or {} if absent / unreadable.

    Best-effort: never raises. Used only for figure-title metadata, so
    missing/corrupt config is a soft-fail (titles just drop the suffix).
    """
    config_path = Path(run_dir) / "run_config.json"
    try:
        with config_path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.debug("run_config.json unreadable at %s: %s", config_path, exc)
        return {}


def short_run_label(run_config: dict, run_dir: Path) -> str:
    """Build a short label for figure titles.

    Format: '<run-dir-basename> @ <git_sha[:8]> [dirty]' if git provenance
    is present, else just the basename.
    """
    base = Path(run_dir).name or str(run_dir)
    provenance = (run_config or {}).get("provenance", {}) or {}
    sha = provenance.get("git_sha")
    dirty = provenance.get("git_dirty")
    if not sha:
        return base
    short_sha = str(sha)[:8]
    suffix = f" @ {short_sha}"
    if dirty == "dirty":
        suffix += " [dirty]"
    return f"{base}{suffix}"


def iter_country_dirs(run_dir: Path) -> Iterator[Tuple[str, Path]]:
    """Yield (country_name, country_dir) for each per-country dir in run_dir.

    Skips:
    - Non-directories
    - The 'plots' output dir
    - Directories without a '<country>_measurements.csv' file

    country_name is the directory name with underscores converted to
    spaces (inverse of Phase 02.1 D-07's `country.replace(" ", "_")`).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "plots":
            continue
        measurements_csv = child / f"{child.name}_measurements.csv"
        if not measurements_csv.is_file():
            continue
        country_name = child.name.replace("_", " ")
        yield country_name, child


def ensure_out_dir(
    run_dir: Path,
    out_override: Optional[Path],
    subdir: Optional[str] = None,
) -> Path:
    """Resolve and create the output directory for a script.

    - If out_override is set: return out_override (mkdir parents).
    - Else if subdir is set: return <run_dir>/<subdir>/plots.
    - Else: return <run_dir>/plots.
    Always mkdirs the resolved path.
    """
    if out_override is not None:
        out = Path(out_override)
    elif subdir is not None:
        out = Path(run_dir) / subdir / "plots"
    else:
        out = Path(run_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def savefig(fig: "matplotlib.figure.Figure", out_dir: Path, name: str) -> Path:
    """Save fig as <out_dir>/<name>.png and close the figure.

    Always prints the saved path to stdout so the CLI user sees progress.
    Returns the saved Path.
    """
    out_path = Path(out_dir) / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    return out_path
