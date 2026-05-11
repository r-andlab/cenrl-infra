"""Join per-country measurement CSVs against a ground-truth blocklist and plot accuracy.

For each per-country dir under <run-dir>, emits three PNGs into
<country>/plots/ (or <out>/<country>/ if --out is given):
  - accuracy_summary.png     precision/recall/F1/accuracy bars (full run)
  - confusion_matrix.png     2x2 heatmap (tn/fp/fn/tp)
  - f1_over_time.png         rolling precision/recall/F1 vs step_idx

Ground-truth format (sniffed on first line):
  - CSV with header `domain`, one domain per row (matches inputs/gfwatch/,
    inputs/russia/, inputs/kazakhstan/). Leading `*.` wildcards are stripped.
  - Plain text, one domain per line, blank lines and `#`-prefixed lines ignored.
All comparisons are case-insensitive.

Example:
    python3 Infrastructure/scripts/plotting/plot_accuracy.py --run-dir outputs/run1 \\
        --ground-truth inputs/gfwatch/gfwatch-blocklist.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Infrastructure.scripts.plotting._style import (
    apply_paper_style,
    ensure_out_dir,
    iter_country_dirs,
    load_run_config,
    savefig,
    short_run_label,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot accuracy (precision/recall/F1) per country vs a ground-truth blocklist.",
    )
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Orchestrator output directory (must contain <country>/<country>_measurements.csv).",
    )
    p.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to the ground-truth blocklist (CSV with 'domain' header, or one-target-per-line text).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override base output directory. Per-country subdirs are created underneath.",
    )
    p.add_argument(
        "--rolling",
        type=int,
        default=200,
        help="Rolling window for time-series precision/recall/F1. Default 200.",
    )
    return p.parse_args()


def load_ground_truth(path: Path) -> set[str]:
    """Load a ground-truth blocklist into a lowercased set of bare domains.

    Sniff format on the first non-empty line:
      - If it equals 'domain' (case-insensitive), treat as CSV.
      - Otherwise, treat as plain text (one entry per line, # comments and blanks skipped).

    Strips leading `*.` (Russia-style wildcard prefix) so wildcard rows match the
    bare domain. Returns a set of lowercased strings.
    """
    path = Path(path)
    entries: list[str] = []

    # Find the first non-empty line for sniffing.
    first_line: str | None = None
    with path.open("r") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                first_line = stripped
                break

    if first_line is None:
        return set()

    if first_line.lower() == "domain":
        df = pd.read_csv(path)
        if "domain" not in df.columns:
            return set()
        entries = [str(x) for x in df["domain"].dropna().tolist()]
    else:
        with path.open("r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append(line)

    out: set[str] = set()
    for e in entries:
        s = e.strip()
        if s.startswith("*."):
            s = s[2:]
        if s:
            out.add(s.lower())
    return out


def _confusion_counts(blocked: pd.Series, expected: pd.Series) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) as ints."""
    b = blocked.astype(int)
    e = expected.astype(int)
    tp = int(((b == 1) & (e == 1)).sum())
    fp = int(((b == 1) & (e == 0)).sum())
    fn = int(((b == 0) & (e == 1)).sum())
    tn = int(((b == 0) & (e == 0)).sum())
    return tp, fp, fn, tn


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def _safe_plot(name: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        print(f"  ERROR plotting {name}: {exc}", file=sys.stderr)


def _plot_accuracy_summary(metrics: dict, country: str, label: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    keys = ["precision", "recall", "f1", "accuracy"]
    vals = [metrics[k] for k in keys]
    bars = ax.bar(keys, vals, color=sns.color_palette("deep", n_colors=4))
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(f"{country}: accuracy summary - {label}")
    for bar, v in zip(bars, vals):
        ax.annotate(
            f"{v:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize="small",
        )
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "accuracy_summary")


def _plot_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int, country: str, label: str, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=["pred not blocked", "pred blocked"],
        yticklabels=["actual not blocked", "actual blocked"],
        cmap="Blues",
        ax=ax,
        cbar=True,
    )
    ax.set_title(f"{country}: confusion matrix - {label}")
    fig.tight_layout()
    savefig(fig, out_dir, "confusion_matrix")


def _plot_f1_over_time(
    df: pd.DataFrame,
    expected: pd.Series,
    country: str,
    label: str,
    window: int,
    out_dir: Path,
) -> None:
    if len(df) < window:
        print(
            f"  skipping f1_over_time for {country}: only {len(df)} rows, window={window}"
        )
        return

    # Ensure deterministic order along step_idx for the rolling metrics.
    sorted_idx = df["step_idx"].argsort(kind="stable")
    steps = df["step_idx"].to_numpy()[sorted_idx]
    b = df["blocked"].astype(int).to_numpy()[sorted_idx]
    e = expected.astype(int).to_numpy()[sorted_idx]

    tp_seq = (b == 1) & (e == 1)
    fp_seq = (b == 1) & (e == 0)
    fn_seq = (b == 0) & (e == 1)

    s_tp = pd.Series(tp_seq.astype(int)).rolling(window=window, min_periods=window).sum()
    s_fp = pd.Series(fp_seq.astype(int)).rolling(window=window, min_periods=window).sum()
    s_fn = pd.Series(fn_seq.astype(int)).rolling(window=window, min_periods=window).sum()

    denom_p = (s_tp + s_fp).replace(0, np.nan)
    denom_r = (s_tp + s_fn).replace(0, np.nan)
    precision = (s_tp / denom_p).fillna(0.0)
    recall = (s_tp / denom_r).fillna(0.0)
    denom_f = (precision + recall).replace(0, np.nan)
    f1 = (2 * precision * recall / denom_f).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, precision, label="precision", linewidth=1.2)
    ax.plot(steps, recall, label="recall", linewidth=1.2)
    ax.plot(steps, f1, label="f1", linewidth=1.6)
    ax.set_xlabel("step_idx")
    ax.set_ylabel(f"score (rolling, w={window})")
    ax.set_ylim(0, 1)
    ax.set_title(f"{country}: precision/recall/F1 over time - {label}")
    ax.legend(loc="best")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "f1_over_time")


def _resolve_country_out(run_dir: Path, country_dir: Path, out_override: Path | None) -> Path:
    if out_override is not None:
        target = Path(out_override) / country_dir.name
        target.mkdir(parents=True, exist_ok=True)
        return target
    return ensure_out_dir(run_dir, None, subdir=country_dir.name)


def _plot_country(
    country: str,
    country_dir: Path,
    blocked_set: set[str],
    label: str,
    out_override: Path | None,
    run_dir: Path,
    rolling: int,
) -> None:
    csv_path = country_dir / f"{country_dir.name}_measurements.csv"
    df = pd.read_csv(csv_path, parse_dates=["utc_timestamp"])
    if len(df) == 0:
        print(f"skipping {country}: no data in {csv_path.name}")
        return

    expected = df["target"].astype(str).str.lower().isin(blocked_set).astype(int)
    df = df.copy()
    df["blocked"] = df["blocked"].astype(int)

    tp, fp, fn, tn = _confusion_counts(df["blocked"], expected)
    metrics = _metrics(tp, fp, fn, tn)

    country_out = _resolve_country_out(run_dir, country_dir, out_override)
    print(
        f"Plotting {country} ({len(df)} rows) -> {country_out} | "
        f"tp={tp} fp={fp} fn={fn} tn={tn} f1={metrics['f1']:.3f}"
    )

    _safe_plot("accuracy_summary", _plot_accuracy_summary, metrics, country, label, country_out)
    _safe_plot(
        "confusion_matrix",
        _plot_confusion_matrix,
        tp, fp, fn, tn, country, label, country_out,
    )
    _safe_plot(
        "f1_over_time",
        _plot_f1_over_time,
        df, expected, country, label, rolling, country_out,
    )


def main() -> int:
    apply_paper_style()
    args = _parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: --run-dir {run_dir} is not a directory", file=sys.stderr)
        return 2

    blocked_set = load_ground_truth(args.ground_truth)
    if not blocked_set:
        print(
            f"ERROR: ground-truth file {args.ground_truth} produced no entries. "
            "Check the format (CSV with 'domain' header, or one-per-line text).",
            file=sys.stderr,
        )
        return 2

    print(f"Loaded {len(blocked_set)} ground-truth entries from {args.ground_truth}")

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
                blocked_set,
                label,
                args.out,
                run_dir,
                args.rolling,
            )
        except Exception as exc:
            print(f"ERROR processing {country}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
