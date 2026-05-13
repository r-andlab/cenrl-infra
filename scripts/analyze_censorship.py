"""Post-hoc censorship summary for a single CenRL run directory.

Usage:
    python3 scripts/analyze_censorship.py <run_dir>

Scans every per-country CSV at ``<run_dir>/<Country>/<Country>.csv`` (the
layout emitted by both the controlled-evaluation and Infrastructure modes)
and produces a table of every ``(country, category)`` pair where at least
one ``is_blocked=1`` row was observed.

Outputs:
    * ``<run_dir>/censorship_summary.csv`` — always written. Header is::

          country,category,first_detection_episode,first_detection_step,
          total_blocks,total_tests,distinct_blocked_targets

      Rows are sorted by (country ASC, first_detection_episode ASC,
      first_detection_step ASC). When no detections exist anywhere in the
      run, the file is header-only.
    * A Markdown-formatted table on stdout mirroring the CSV. The category
      column may be truncated in the Markdown view for readability; the CSV
      remains untruncated.

Per-country CSV schema (verified):
    episode,time,action,targets,rewards,q_value,is_blocked,is_optimal,coverage

    * ``action`` is ``"categories <Category Name>"``. Rows whose ``action``
      does not start with ``"categories "`` are skipped defensively so the
      script does not crash on mixed-feature runs.
    * Ordering for "first detection" is (episode ASC, time ASC).

This script is a leaf utility: stdlib only, no imports from ``models/``,
``Infrastructure/``, ``common/``, ``api/``, or ``baselines/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

CATEGORY_PREFIX = "categories "
CSV_HEADER = [
    "country",
    "category",
    "first_detection_episode",
    "first_detection_step",
    "total_blocks",
    "total_tests",
    "distinct_blocked_targets",
]
# Stdout-only truncation cap for the category column.
STDOUT_CATEGORY_MAX = 40


@dataclass
class PairAggregator:
    """Running counts for a single (country, category) pair."""

    total_tests: int = 0
    total_blocks: int = 0
    distinct_blocked_targets: set[str] = field(default_factory=set)
    # (episode, time) of the lexicographically first is_blocked=1 row.
    first_block: tuple[int, int] | None = None


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
            category = action[len(CATEGORY_PREFIX):]
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
                    agg.distinct_blocked_targets.add(target)
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
                "first_detection_episode": episode,
                "first_detection_step": step,
                "total_blocks": agg.total_blocks,
                "total_tests": agg.total_tests,
                "distinct_blocked_targets": len(agg.distinct_blocked_targets),
            }
        )
    rows.sort(
        key=lambda r: (
            r["country"],
            r["first_detection_episode"],
            r["first_detection_step"],
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
                str(row["first_detection_episode"]),
                str(row["first_detection_step"]),
                str(row["total_blocks"]),
                str(row["total_tests"]),
                str(row["distinct_blocked_targets"]),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
