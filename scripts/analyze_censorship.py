"""Post-hoc censorship summary for a single CenRL run directory.

Usage:
    python3 scripts/analyze_censorship.py <run_dir>

Scans every per-country CSV at ``<run_dir>/<Country>/<Country>.csv`` (the
layout emitted by both the controlled-evaluation and Infrastructure modes)
and produces a table of every ``(country, category)`` pair where at least
one ``is_blocked=1`` row was observed.

Outputs:
    * ``<run_dir>/censorship_summary.csv`` — always written. Header is::

          country,category,first_det_ep,first_det_step,
          total_blocks,total_tests,distinct

      Rows are sorted by (country ASC, first_det_ep ASC,
      first_det_step ASC). When no detections exist anywhere in the
      run, the file is header-only.
    * A Markdown-formatted table on stdout mirroring the CSV. The category
      column may be truncated in the Markdown view for readability; the CSV
      remains untruncated.
    * ``<run_dir>/censorship_summary.png`` — a matplotlib-rendered table with
      a title block above (run name, date, measurements per episode, total
      countries, total episodes). When the row count exceeds ``--max-rows``
      (default 60), the PNG shows the top-N pairs by ``total_blocks`` and a
      footer notes the truncation; the CSV is always the full data.

Per-country CSV schema (verified):
    episode,time,action,targets,rewards,q_value,is_blocked,is_optimal,coverage

    * ``action`` is ``"categories <Category Name>"``. Rows whose ``action``
      does not start with ``"categories "`` are skipped defensively so the
      script does not crash on mixed-feature runs.
    * Ordering for "first detection" is (episode ASC, time ASC).

This script depends only on the stdlib plus matplotlib (already a project
dep). It does not import from ``models/``, ``Infrastructure/``, ``common/``,
``api/``, or ``baselines/``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

CATEGORY_PREFIX = "categories "
CSV_HEADER = [
    "country",
    "category",
    "first_det_ep",
    "first_det_step",
    "total_blocks",
    "total_tests",
    "distinct",
]
# Stdout-only truncation cap for the category column.
STDOUT_CATEGORY_MAX = 40


@dataclass
class PairAggregator:
    """Running counts for a single (country, category) pair."""

    total_tests: int = 0
    total_blocks: int = 0
    distinct: set[str] = field(default_factory=set)
    # (episode, time) of the lexicographically first is_blocked=1 row.
    first_block: tuple[int, int] | None = None


@dataclass
class RunMeta:
    """Run-level metadata surfaced in the PNG title block."""

    run_name: str
    date: str
    measurements_per_episode: int
    total_countries: int
    total_episodes: int


def discover_country_csvs(run_dir: Path) -> list[tuple[str, Path]]:
    """Find ``<Country>/<Country>.csv`` files under ``run_dir``.

    Returns a list of ``(country_label, csv_path)`` tuples. ``country_label``
    is the subdirectory name verbatim (underscores kept).
    """
    found: list[tuple[str, Path]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / f"{child.name}.csv"
        if candidate.is_file():
            found.append((child.name, candidate))
    return found


def aggregate_country(
    country: str,
    csv_path: Path,
    pairs: dict[tuple[str, str], PairAggregator],
) -> None:
    """Stream ``csv_path`` and update ``pairs`` for the given country."""
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            action = row.get("action", "")
            if not action.startswith(CATEGORY_PREFIX):
                # Defensive: ignore non-categories rows (e.g. other feature
                # modes in mixed runs) rather than crashing.
                continue
            category = action[len(CATEGORY_PREFIX) :]
            key = (country, category)
            agg = pairs.get(key)
            if agg is None:
                agg = PairAggregator()
                pairs[key] = agg
            agg.total_tests += 1

            if row.get("is_blocked") == "1":
                try:
                    episode = int(row["episode"])
                    step = int(row["time"])
                except (KeyError, ValueError):
                    # Malformed row — skip the detection bookkeeping but
                    # leave total_tests bumped so the per-pair denominator
                    # still reflects what was read.
                    continue
                agg.total_blocks += 1
                target = row.get("targets", "")
                if target:
                    agg.distinct.add(target)
                candidate = (episode, step)
                if agg.first_block is None or candidate < agg.first_block:
                    agg.first_block = candidate


def build_rows(
    pairs: dict[tuple[str, str], PairAggregator],
) -> list[dict[str, object]]:
    """Filter to pairs with >=1 detection and emit sorted output rows."""
    rows: list[dict[str, object]] = []
    for (country, category), agg in pairs.items():
        if agg.total_blocks == 0 or agg.first_block is None:
            continue
        episode, step = agg.first_block
        rows.append(
            {
                "country": country,
                "category": category,
                "first_det_ep": episode,
                "first_det_step": step,
                "total_blocks": agg.total_blocks,
                "total_tests": agg.total_tests,
                "distinct": len(agg.distinct),
            }
        )
    rows.sort(
        key=lambda r: (
            r["country"],
            r["first_det_ep"],
            r["first_det_step"],
        )
    )
    return rows


def write_csv(out_path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write ``rows`` to ``out_path`` (always with header)."""
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        # Use LF (not CRLF) so downstream Unix tooling (grep, awk anchored
        # with $) matches on the trailing field without stripping \r first.
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row[col] for col in CSV_HEADER])


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "\u2026"


def render_markdown(rows: list[dict[str, object]]) -> str:
    """Render ``rows`` as a Markdown table. Category truncated for display."""
    display_rows: list[list[str]] = []
    for row in rows:
        display_rows.append(
            [
                str(row["country"]),
                _truncate(str(row["category"]), STDOUT_CATEGORY_MAX),
                str(row["first_det_ep"]),
                str(row["first_det_step"]),
                str(row["total_blocks"]),
                str(row["total_tests"]),
                str(row["distinct"]),
            ]
        )

    widths = [len(h) for h in CSV_HEADER]
    for dr in display_rows:
        for i, cell in enumerate(dr):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def fmt_row(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))
            + " |"
        )

    lines = [
        fmt_row(list(CSV_HEADER)),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for dr in display_rows:
        lines.append(fmt_row(dr))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run-level metadata + PNG rendering
# ---------------------------------------------------------------------------

# Match the leading timestamp in app.log lines of the form
# "2026-04-27 12:55:22,052 - INFO - ..."  — keep date+time only.
_APP_LOG_TS_LEN = len("YYYY-MM-DD HH:MM:SS")


def _read_app_log_timestamp(run_dir: Path) -> str | None:
    """Best-effort: return the YYYY-MM-DD date from app.log line 1, else None."""
    log_path = run_dir / "app.log"
    if not log_path.is_file():
        return None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return None
    if len(first) < _APP_LOG_TS_LEN:
        return None
    candidate = first[:_APP_LOG_TS_LEN]
    try:
        dt.datetime.strptime(candidate, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Fall back to just the date portion if the time part is malformed.
        try:
            dt.datetime.strptime(candidate[:10], "%Y-%m-%d")
            return candidate[:10]
        except ValueError:
            return None
    return candidate


def _mtime_date(run_dir: Path) -> str:
    """Fallback date source: the run directory's mtime as YYYY-MM-DD HH:MM:SS."""
    ts = dt.datetime.fromtimestamp(run_dir.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def collect_run_meta(run_dir: Path, country_csvs: list[tuple[str, Path]]) -> RunMeta:
    """Build a ``RunMeta`` by scanning ``country_csvs`` for episode/step caps.

    ``measurements_per_episode`` is ``max(time)`` observed across all rows
    and episodes (CenRL caps ``time`` at the per-episode batch size, so the
    max value is the canonical batch size for the run). If no data rows
    exist, both episode-count and measurements default to 0.
    """
    date = _read_app_log_timestamp(run_dir) or _mtime_date(run_dir)

    max_step = 0
    episodes_seen: set[int] = set()
    for _country, csv_path in country_csvs:
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        ep = int(row["episode"])
                        st = int(row["time"])
                    except (KeyError, ValueError):
                        continue
                    episodes_seen.add(ep)
                    if st > max_step:
                        max_step = st
        except OSError:
            # If a single per-country file is unreadable, skip it — the main
            # aggregation pass will surface the failure if it matters.
            continue

    return RunMeta(
        run_name=run_dir.name,
        date=date,
        measurements_per_episode=max_step,
        total_countries=len(country_csvs),
        total_episodes=len(episodes_seen),
    )


# Default PNG row cap. matplotlib's table renderer degrades badly past a few
# hundred rows on a single figure; 60 keeps the PNG legible and the CSV
# remains the source of truth for full data.
_DEFAULT_MAX_PNG_ROWS = 60


def _select_png_rows(
    rows: list[dict[str, object]], max_rows: int
) -> tuple[list[dict[str, object]], bool]:
    """Return (rows_to_render, truncated). ``max_rows<=0`` means unlimited.

    When truncated, the top ``max_rows`` pairs by ``total_blocks`` DESC are
    selected (ties broken by ``total_tests`` DESC, then country/category for
    determinism), then re-sorted into the canonical display order so the
    PNG ordering matches the CSV.
    """
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows, False
    ranked = sorted(
        rows,
        key=lambda r: (
            -int(r["total_blocks"]),
            -int(r["total_tests"]),
            str(r["country"]),
            str(r["category"]),
        ),
    )
    picked = ranked[:max_rows]
    picked.sort(
        key=lambda r: (
            r["country"],
            r["first_det_ep"],
            r["first_det_step"],
        )
    )
    return picked, True


def render_png(
    out_path: Path,
    rows: list[dict[str, object]],
    meta: RunMeta,
    max_rows: int = _DEFAULT_MAX_PNG_ROWS,
) -> None:
    """Render a PNG table with a metadata title block above the rows.

    Uses ``matplotlib.pyplot.table`` with mathtext only (no system LaTeX
    dependency). When ``rows`` is empty the PNG still renders, showing only
    the title block and a "No censorship detected" note.
    """
    # Import lazily so the CSV-only path still works on minimal environments
    # without matplotlib installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_rows, truncated = _select_png_rows(rows, max_rows)
    total_pairs = len(rows)
    shown_pairs = len(display_rows)

    title_lines = [
        rf"$\bf{{Run:}}$ {meta.run_name}",
        rf"$\bf{{Date:}}$ {meta.date}",
        (
            rf"$\bf{{Measurements/episode:}}$ {meta.measurements_per_episode}"
            rf"   $\bf{{Episodes:}}$ {meta.total_episodes}"
            rf"   $\bf{{Countries:}}$ {meta.total_countries}"
        ),
    ]
    title_text = "\n".join(title_lines)

    # Cell strings — use the same Markdown truncation rule for the category
    # column so very long category names do not blow out the figure width.
    headers = list(CSV_HEADER)
    body_cells: list[list[str]] = []
    for r in display_rows:
        body_cells.append(
            [
                str(r["country"]),
                _truncate(str(r["category"]), STDOUT_CATEGORY_MAX),
                str(r["first_det_ep"]),
                str(r["first_det_step"]),
                str(r["total_blocks"]),
                str(r["total_tests"]),
                str(r["distinct"]),
            ]
        )

    # Heuristic figure sizing: 11 in wide, ~0.32 in per body row, + title/footer.
    fig_width = 11.0
    row_height = 0.32
    title_height = 1.1
    footer_height = 0.6 if truncated or shown_pairs == 0 else 0.25
    body_height = max(shown_pairs, 1) * row_height + 0.6  # +header allowance
    fig_height = title_height + body_height + footer_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # Title block: anchored at the top of the axes.
    ax.text(
        0.0,
        1.0,
        title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        family="DejaVu Sans",
    )

    if shown_pairs == 0:
        ax.text(
            0.5,
            0.5,
            "No censorship detected in this run.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#555555",
        )
    else:
        # Reserve roughly the top (title_height / fig_height) of the axes
        # for the title block, place the table below it.
        table_top = 1.0 - (title_height / fig_height)
        table = ax.table(
            cellText=body_cells,
            colLabels=headers,
            loc="upper left",
            bbox=[0.0, 0.05, 1.0, table_top - 0.05],
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        # Shrink font when many rows.
        font_size = 9 if shown_pairs <= 40 else 7
        table.set_fontsize(font_size)

        # Style header row + zebra-stripe body rows for readability.
        n_cols = len(headers)
        header_face = "#d9d9d9"
        zebra = "#f4f4f4"
        for col_idx in range(n_cols):
            cell = table[0, col_idx]
            cell.set_facecolor(header_face)
            cell.set_text_props(weight="bold")
        for r_idx in range(1, shown_pairs + 1):
            for col_idx in range(n_cols):
                cell = table[r_idx, col_idx]
                if r_idx % 2 == 0:
                    cell.set_facecolor(zebra)

    # Footer: truncation note (if applicable) plus a CSV pointer.
    footer_parts: list[str] = []
    if truncated:
        footer_parts.append(
            f"Showing top {shown_pairs} of {total_pairs} pairs by total_blocks. "
            "Full data: censorship_summary.csv"
        )
    elif total_pairs > 0:
        footer_parts.append(f"{total_pairs} (country, category) pairs with censorship.")
    if footer_parts:
        ax.text(
            0.0,
            0.0,
            "\n".join(footer_parts),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise per-(country, category) censorship detections for a "
            "single CenRL run directory."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run directory (e.g. outputs/outtest7).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=_DEFAULT_MAX_PNG_ROWS,
        help=(
            "Cap rows in the PNG table (default %(default)s). Use 0 for "
            "unlimited. CSV output is never capped."
        ),
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG rendering (CSV + stdout only).",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.exists() or not run_dir.is_dir():
        print(
            f"error: run directory does not exist or is not a directory: {run_dir}",
            file=sys.stderr,
        )
        return 1

    country_csvs = discover_country_csvs(run_dir)
    if not country_csvs:
        print(
            f"error: no per-country CSVs found under {run_dir} "
            "(expected <run_dir>/<Country>/<Country>.csv)",
            file=sys.stderr,
        )
        return 1

    pairs: dict[tuple[str, str], PairAggregator] = defaultdict(PairAggregator)
    # defaultdict is convenient, but we want to reuse the same instances
    # across calls and rely on identity; cast back to a plain dict view by
    # passing it through aggregate_country.
    for country, csv_path in country_csvs:
        aggregate_country(country, csv_path, pairs)

    rows = build_rows(pairs)
    out_path = run_dir / "censorship_summary.csv"
    write_csv(out_path, rows)

    if rows:
        print(render_markdown(rows))
    else:
        print(f"No censorship detected in {run_dir}.")

    if not args.no_png:
        meta = collect_run_meta(run_dir, country_csvs)
        png_path = run_dir / "censorship_summary.png"
        render_png(png_path, rows, meta, max_rows=args.max_rows)
        print(f"Wrote PNG: {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
