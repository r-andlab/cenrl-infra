"""Visualize per-country <country>_measurements.csv files for learning/quality analysis.

For each per-country dir under <run-dir>, emits PNGs into
<country>/plots/ (or <out>/<country>/ if --out is given). Per-step charts emit
two variants each (see ``## Notes`` in the README): ``*_monotonic.png`` plots
the run as one continuous trajectory ordered by ``utc_timestamp``, and
``*_episode_mean.png`` aggregates across episodes by ``step_idx`` with a
mean +/- std band.

  - reward_cumulative_monotonic.png       cumulative reward vs measurement_idx
  - reward_cumulative_episode_mean.png    per-episode cumsum mean +/- std across episodes
  - reward_rolling_monotonic.png          rolling-mean reward vs measurement_idx
  - reward_rolling_episode_mean.png       reward mean +/- std across episodes by step_idx
  - blocked_rate_rolling_monotonic.png    rolling-mean blocked rate vs measurement_idx
  - blocked_rate_rolling_episode_mean.png blocked mean +/- std across episodes by step_idx
  - latency_distribution.png              latency_ms histogram + KDE
  - vp_count_distribution.png             vp_count discrete histogram
  - coverage_over_time_monotonic.png      coverage vs measurement_idx
  - coverage_over_time_episode_mean.png   coverage mean +/- std across episodes by step_idx
  - optimal_arm_rate_monotonic.png        rolling-mean is_optimal vs measurement_idx
  - optimal_arm_rate_episode_mean.png     is_optimal mean +/- std across episodes by step_idx
  - q_value_top_arms_monotonic.png        q_value vs measurement_idx for top-N arms
  - q_value_top_arms_episode_mean.png     q_value mean +/- std across episodes for top-N arms
  - arm_counts.png                        bar chart of arm visit counts (top-N + other)

Example:
    python3 Infrastructure/scripts/plotting/plot_measurements.py --run-dir outputs/run1 \\
        --top-arms 10 --rolling 100
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
    add_monotonic_index,
    apply_paper_style,
    ensure_out_dir,
    episode_mean_band,
    iter_country_dirs,
    load_run_config,
    savefig,
    short_run_label,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-country measurement CSVs from a Phase 3 run directory.",
    )
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Orchestrator output directory (must contain <country>/<country>_measurements.csv).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Override base output directory. With --out, per-country plots go to "
            "<out>/<country>/. Without --out, per-country plots go to "
            "<run-dir>/<country>/plots/."
        ),
    )
    p.add_argument(
        "--top-arms",
        type=int,
        default=10,
        help="How many arms (by visit count) to show in q_value_top_arms_*.png and arm_counts.png. Default 10.",
    )
    p.add_argument(
        "--rolling",
        type=int,
        default=100,
        help="Rolling-mean window for reward, blocked rate, and optimal-arm rate. Default 100.",
    )
    return p.parse_args()


def _resolve_country_out(run_dir: Path, country_dir: Path, out_override: Path | None) -> Path:
    """Resolve where this country's plots go.

    - If --out is set: out_override / country_dir.name (no extra 'plots' subdir;
      the override IS the user's chosen base).
    - Else: <country_dir>/plots/
    """
    if out_override is not None:
        target = Path(out_override) / country_dir.name
        target.mkdir(parents=True, exist_ok=True)
        return target
    return ensure_out_dir(run_dir, None, subdir=country_dir.name)


def _safe_plot(name: str, fn, *args, **kwargs) -> None:
    """Wrap a single figure emission so one failure doesn't kill the country."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        print(f"  ERROR plotting {name}: {exc}", file=sys.stderr)


def _plot_reward_cumulative_monotonic(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    df_m = add_monotonic_index(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_m["measurement_idx"], df_m["reward"].cumsum(), linewidth=1.2)
    ax.set_xlabel("measurement_idx (time-ordered)")
    ax.set_ylabel("cumulative reward")
    ax.set_title(f"{country}: cumulative reward (monotonic) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "reward_cumulative_monotonic")


def _plot_reward_cumulative_episode_mean(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    if "episode_idx" not in df.columns or "step_idx" not in df.columns:
        return
    # cumsum WITHIN each episode, then aggregate across episodes by step_idx
    per_ep = df.copy()
    per_ep["reward_cumsum"] = per_ep.groupby("episode_idx")["reward"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 5))
    episode_mean_band(ax, per_ep, "step_idx", "reward_cumsum")
    ax.set_xlabel("step_idx (per-episode)")
    ax.set_ylabel("cumulative reward (mean +/- std across episodes)")
    ax.set_title(f"{country}: cumulative reward (episode mean) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "reward_cumulative_episode_mean")


def _plot_reward_rolling_monotonic(df: pd.DataFrame, country: str, label: str, window: int, out_dir: Path) -> None:
    df_m = add_monotonic_index(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df_m["measurement_idx"],
        df_m["reward"].rolling(window=window, min_periods=1).mean(),
        linewidth=1.2,
    )
    ax.set_xlabel("measurement_idx (time-ordered)")
    ax.set_ylabel(f"reward (rolling mean, w={window})")
    ax.set_title(f"{country}: rolling reward (monotonic) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "reward_rolling_monotonic")


def _plot_reward_rolling_episode_mean(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    if "episode_idx" not in df.columns or "step_idx" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    episode_mean_band(ax, df, "step_idx", "reward")
    ax.set_xlabel("step_idx (per-episode)")
    ax.set_ylabel("reward (mean +/- std across episodes)")
    ax.set_title(f"{country}: reward (episode mean) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "reward_rolling_episode_mean")


def _plot_blocked_rate_monotonic(df: pd.DataFrame, country: str, label: str, window: int, out_dir: Path) -> None:
    df_m = add_monotonic_index(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df_m["measurement_idx"],
        df_m["blocked"].rolling(window=window, min_periods=1).mean(),
        linewidth=1.2,
    )
    ax.set_xlabel("measurement_idx (time-ordered)")
    ax.set_ylabel(f"blocked rate (rolling mean, w={window})")
    ax.set_ylim(0, 1)
    ax.set_title(f"{country}: rolling blocked rate (monotonic) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "blocked_rate_rolling_monotonic")


def _plot_blocked_rate_episode_mean(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    if "episode_idx" not in df.columns or "step_idx" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    episode_mean_band(ax, df, "step_idx", "blocked")
    ax.set_xlabel("step_idx (per-episode)")
    ax.set_ylabel("blocked rate (mean +/- std across episodes)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{country}: blocked rate (episode mean) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "blocked_rate_rolling_episode_mean")


def _plot_latency_distribution(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    data = df["latency_ms"].dropna()
    if len(data) == 0:
        print(f"  skipping latency_distribution for {country}: latency_ms all NaN")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        sns.histplot(data, kde=True, bins=50, ax=ax)
    except Exception as exc:
        print(f"  warn: histplot kde failed for latency_ms: {exc}; retry without kde")
        ax.cla()
        sns.histplot(data, kde=False, bins=50, ax=ax)
    ax.set_xlabel("latency_ms (schedule -> absorb)")
    ax.set_title(f"{country}: latency distribution - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "latency_distribution")


def _plot_vp_count_distribution(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    data = df["vp_count"].dropna()
    if len(data) == 0:
        print(f"  skipping vp_count_distribution for {country}: vp_count all NaN")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data, discrete=True, ax=ax)
    ax.set_xlabel("vp_count (VPs voting in aggregated result)")
    ax.set_title(f"{country}: vp_count distribution - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "vp_count_distribution")


def _plot_coverage_over_time_monotonic(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    df_m = add_monotonic_index(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_m["measurement_idx"], df_m["coverage"], linewidth=1.0)
    ax.set_xlabel("measurement_idx (time-ordered)")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1)
    ax.set_title(f"{country}: coverage over time (monotonic) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "coverage_over_time_monotonic")


def _plot_coverage_over_time_episode_mean(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    if "episode_idx" not in df.columns or "step_idx" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    episode_mean_band(ax, df, "step_idx", "coverage")
    ax.set_xlabel("step_idx (per-episode)")
    ax.set_ylabel("coverage (mean +/- std across episodes)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{country}: coverage over time (episode mean) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "coverage_over_time_episode_mean")


def _plot_optimal_arm_rate_monotonic(df: pd.DataFrame, country: str, label: str, window: int, out_dir: Path) -> None:
    if "is_optimal" not in df.columns:
        print(f"  skipping optimal_arm_rate_monotonic for {country}: is_optimal column missing")
        return
    df_m = add_monotonic_index(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df_m["measurement_idx"],
        df_m["is_optimal"].rolling(window=window, min_periods=1).mean(),
        linewidth=1.2,
    )
    ax.set_xlabel("measurement_idx (time-ordered)")
    ax.set_ylabel(f"optimal-arm rate (rolling, w={window})")
    ax.set_ylim(0, 1)
    ax.set_title(f"{country}: optimal arm rate (monotonic) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "optimal_arm_rate_monotonic")


def _plot_optimal_arm_rate_episode_mean(df: pd.DataFrame, country: str, label: str, out_dir: Path) -> None:
    if "is_optimal" not in df.columns:
        print(f"  skipping optimal_arm_rate_episode_mean for {country}: is_optimal column missing")
        return
    if "episode_idx" not in df.columns or "step_idx" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    episode_mean_band(ax, df, "step_idx", "is_optimal")
    ax.set_xlabel("step_idx (per-episode)")
    ax.set_ylabel("optimal-arm rate (mean +/- std across episodes)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{country}: optimal arm rate (episode mean) - {label}")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "optimal_arm_rate_episode_mean")


def _plot_q_value_top_arms_monotonic(
    df: pd.DataFrame, country: str, label: str, top_n: int, top_arms: list, out_dir: Path
) -> None:
    top_df = df[df["arm"].isin(top_arms)]
    if len(top_df) == 0:
        print(f"  skipping q_value_top_arms_monotonic for {country}: no rows after top-N filter")
        return
    df_m_top = add_monotonic_index(top_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_m_top, x="measurement_idx", y="q_value", hue="arm", ax=ax)
    ax.set_xlabel("measurement_idx (time-ordered)")
    ax.set_ylabel("q_value")
    ax.set_title(f"{country}: q_value for top-{len(top_arms)} arms (monotonic) - {label}")
    ax.legend(loc="best", fontsize="x-small", ncol=2)
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "q_value_top_arms_monotonic")


def _plot_q_value_top_arms_episode_mean(
    df: pd.DataFrame, country: str, label: str, top_n: int, top_arms: list, out_dir: Path
) -> None:
    top_df = df[df["arm"].isin(top_arms)]
    if len(top_df) == 0:
        print(f"  skipping q_value_top_arms_episode_mean for {country}: no rows after top-N filter")
        return
    if "episode_idx" not in df.columns or "step_idx" not in df.columns:
        return
    palette = sns.color_palette("deep", n_colors=max(len(top_arms), 1))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, arm in enumerate(top_arms):
        arm_df = top_df[top_df["arm"] == arm]
        if len(arm_df) == 0:
            continue
        episode_mean_band(
            ax,
            arm_df,
            "step_idx",
            "q_value",
            label=str(arm),
            color=palette[i % len(palette)],
        )
    ax.set_xlabel("step_idx (per-episode)")
    ax.set_ylabel("q_value (mean +/- std across episodes)")
    ax.set_title(f"{country}: q_value for top-{len(top_arms)} arms (episode mean) - {label}")
    ax.legend(loc="best", fontsize="x-small", ncol=2)
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "q_value_top_arms_episode_mean")


def _plot_arm_counts(df: pd.DataFrame, country: str, label: str, top_n: int, out_dir: Path) -> None:
    arm_counts = df["arm"].value_counts()
    top = arm_counts.head(top_n)
    other_total = int(arm_counts.iloc[top_n:].sum()) if len(arm_counts) > top_n else 0
    labels = list(top.index)
    values = list(top.values)
    if other_total > 0:
        labels.append("other")
        values.append(other_total)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_xlabel("arm")
    ax.set_ylabel("visit count")
    ax.set_title(f"{country}: arm visit counts (top {top_n} + other) - {label}")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "arm_counts")


def _plot_country(
    country: str,
    country_dir: Path,
    label: str,
    out_override: Path | None,
    run_dir: Path,
    top_arms: int,
    rolling: int,
) -> None:
    csv_path = country_dir / f"{country_dir.name}_measurements.csv"
    df = pd.read_csv(csv_path, parse_dates=["utc_timestamp"])
    if len(df) == 0:
        print(f"skipping {country}: no data in {csv_path.name}")
        return

    country_out = _resolve_country_out(run_dir, country_dir, out_override)
    print(f"Plotting {country} ({len(df)} rows) -> {country_out}")

    # Identify top-N arms once from full df so both q_value variants stay aligned.
    top_arm_names = list(df["arm"].value_counts().head(top_arms).index)

    _safe_plot("reward_cumulative_monotonic", _plot_reward_cumulative_monotonic, df, country, label, country_out)
    _safe_plot("reward_cumulative_episode_mean", _plot_reward_cumulative_episode_mean, df, country, label, country_out)
    _safe_plot("reward_rolling_monotonic", _plot_reward_rolling_monotonic, df, country, label, rolling, country_out)
    _safe_plot("reward_rolling_episode_mean", _plot_reward_rolling_episode_mean, df, country, label, country_out)
    _safe_plot("blocked_rate_rolling_monotonic", _plot_blocked_rate_monotonic, df, country, label, rolling, country_out)
    _safe_plot("blocked_rate_rolling_episode_mean", _plot_blocked_rate_episode_mean, df, country, label, country_out)
    _safe_plot("coverage_over_time_monotonic", _plot_coverage_over_time_monotonic, df, country, label, country_out)
    _safe_plot("coverage_over_time_episode_mean", _plot_coverage_over_time_episode_mean, df, country, label, country_out)
    _safe_plot("optimal_arm_rate_monotonic", _plot_optimal_arm_rate_monotonic, df, country, label, rolling, country_out)
    _safe_plot("optimal_arm_rate_episode_mean", _plot_optimal_arm_rate_episode_mean, df, country, label, country_out)
    _safe_plot("q_value_top_arms_monotonic", _plot_q_value_top_arms_monotonic, df, country, label, top_arms, top_arm_names, country_out)
    _safe_plot("q_value_top_arms_episode_mean", _plot_q_value_top_arms_episode_mean, df, country, label, top_arms, top_arm_names, country_out)
    _safe_plot("latency_distribution", _plot_latency_distribution, df, country, label, country_out)
    _safe_plot("vp_count_distribution", _plot_vp_count_distribution, df, country, label, country_out)
    _safe_plot("arm_counts", _plot_arm_counts, df, country, label, top_arms, country_out)


def main() -> int:
    apply_paper_style()
    args = _parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: --run-dir {run_dir} is not a directory", file=sys.stderr)
        return 2

    run_config = load_run_config(run_dir)
    label = short_run_label(run_config, run_dir)

    countries = list(iter_country_dirs(run_dir))
    if not countries:
        print(f"No per-country measurement CSVs found under {run_dir}")
        return 0

    print(f"Found {len(countries)} country dir(s) with measurements.")
    for country, country_dir in countries:
        try:
            _plot_country(
                country,
                country_dir,
                label,
                args.out,
                run_dir,
                args.top_arms,
                args.rolling,
            )
        except Exception as exc:
            print(f"ERROR processing {country}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
