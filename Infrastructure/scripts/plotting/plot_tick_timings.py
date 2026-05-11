"""Visualize <run-dir>/tick_timings.csv for Python-vs-Go bottleneck analysis.

Produces four PNGs under <run-dir>/plots/ (or --out):
  - tick_timings_stacked.png             stacked-area sub-step decomposition
  - tick_timings_total.png               total tick time + rolling mean
  - tick_timings_distributions.png       per-substep histograms (2x3 grid)
  - tick_timings_vs_active_countries.png total tick time vs active country count

Example:
    python3 Infrastructure/scripts/plotting/plot_tick_timings.py --run-dir outputs/run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Infrastructure.scripts.plotting._style import (
    apply_paper_style,
    ensure_out_dir,
    load_run_config,
    savefig,
    short_run_label,
)

SUB_STEPS = ["drain", "eval_processing", "aggregate", "model_update", "schedule"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot tick_timings.csv from a Phase 3 run directory."
    )
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Orchestrator output directory (must contain tick_timings.csv).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output directory. Default: <run-dir>/plots/",
    )
    return p.parse_args()


def _plot_stacked(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    # Build a (n_steps, n_rows) array for stackplot.
    y = np.vstack([df[col].to_numpy(dtype=float) for col in SUB_STEPS])
    ax.stackplot(df["tick_idx"].to_numpy(), y, labels=SUB_STEPS, alpha=0.85)
    ax.set_xlabel("tick_idx")
    ax.set_ylabel("seconds")
    ax.set_title(f"Tick sub-step decomposition - {label}")
    ax.legend(loc="upper right")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "tick_timings_stacked")


def _plot_total(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    window = max(1, len(df) // 50)
    ax.plot(df["tick_idx"], df["total"], label="total", alpha=0.5, linewidth=0.8)
    ax.plot(
        df["tick_idx"],
        df["total"].rolling(window=window, min_periods=1).mean(),
        label=f"rolling mean (w={window})",
        linewidth=1.8,
    )
    ax.set_xlabel("tick_idx")
    ax.set_ylabel("total tick time (s)")
    ax.set_title(f"Total tick time - {label}")
    ax.legend(loc="upper right")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "tick_timings_total")


def _plot_distributions(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    cols = SUB_STEPS + ["total"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), cols):
        data = df[col].dropna()
        if len(data) == 0:
            ax.set_title(f"{col} (no data)")
            ax.axis("off")
            continue
        try:
            sns.histplot(data, kde=True, ax=ax)
        except Exception as exc:  # KDE can fail on constant data
            print(f"  warn: histplot kde failed for {col}: {exc}; retry without kde")
            ax.cla()
            sns.histplot(data, kde=False, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("seconds")
    fig.suptitle(f"Tick sub-step distributions - {label}")
    fig.tight_layout()
    savefig(fig, out_dir, "tick_timings_distributions")


def _plot_vs_active_countries(df: pd.DataFrame, label: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    n_unique = df["n_active_countries"].nunique(dropna=True)
    if n_unique <= 1:
        # Skip the regression fit; just scatter.
        sns.scatterplot(
            data=df,
            x="n_active_countries",
            y="total",
            ax=ax,
            alpha=0.4,
            s=12,
        )
    else:
        try:
            sns.regplot(
                data=df,
                x="n_active_countries",
                y="total",
                ax=ax,
                scatter_kws={"alpha": 0.4, "s": 12},
            )
        except np.linalg.LinAlgError as exc:
            print(f"  warn: regplot LinAlg failed: {exc}; falling back to scatter")
            ax.cla()
            sns.scatterplot(
                data=df,
                x="n_active_countries",
                y="total",
                ax=ax,
                alpha=0.4,
                s=12,
            )
    ax.set_xlabel("n_active_countries (start of tick)")
    ax.set_ylabel("total tick time (s)")
    ax.set_title(f"Tick total vs active country count - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "tick_timings_vs_active_countries")


def main() -> int:
    apply_paper_style()
    args = _parse_args()

    run_dir: Path = args.run_dir
    out_dir = ensure_out_dir(run_dir, args.out)

    csv_path = run_dir / "tick_timings.csv"
    df = pd.read_csv(csv_path, parse_dates=["utc_timestamp"])

    if len(df) == 0:
        print(f"tick_timings.csv is empty (header only) at {csv_path}; nothing to plot.")
        return 0

    run_config = load_run_config(run_dir)
    label = short_run_label(run_config, run_dir)

    print(f"Plotting tick timings for {label} ({len(df)} ticks) -> {out_dir}")

    figure_fns = (
        ("stacked", _plot_stacked),
        ("total", _plot_total),
        ("distributions", _plot_distributions),
        ("vs_active_countries", _plot_vs_active_countries),
    )
    for name, fn in figure_fns:
        try:
            fn(df, label, out_dir)
        except Exception as exc:
            print(f"  ERROR plotting {name}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
