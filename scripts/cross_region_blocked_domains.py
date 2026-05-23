"""Cross-region blocked-domain summary for a single CenRL run directory.

Usage:
    python3 scripts/cross_region_blocked_domains.py <run_dir>

Scans every per-country CSV at ``<run_dir>/<Country>/<Country>.csv`` and
prints a terminal-friendly summary to stdout describing cross-region blocking
patterns. No CSVs, no PNGs, no side-effects — stdout only.

Outputs (in order):
    1. Header line: ``## Cross-region blocked-domain summary: <run_dir.name>``
    2. Three single-line stats:
         - Distinct domains blocked in any region
         - Distinct domains blocked in exactly one region
         - Distinct domains blocked in 3 or more regions
    3. Markdown table: top 10 blocked domains aggregated across regions.
    4. Markdown table: blocked-domain set size per region.
    5. Markdown table: measurement volume per region (with pct_of_total).

Per-country CSV schema (verified):
    episode,time,action,targets,rewards,q_value,is_blocked,is_optimal,coverage

    * A row is "blocked" iff ``is_blocked == "1"`` exactly.
    * "Domain" == ``targets`` (string, treated as valid only when non-empty
      after ``.strip()``).
    * "Region" == the country subdirectory name verbatim (underscores kept,
      e.g. ``United_States``).
    * Unlike ``analyze_censorship.py`` this script does NOT filter by
      ``action`` prefix — every blocked row with a non-empty target counts,
      regardless of which feature drove the action.
    * Total-measurements counts ALL rows in the per-country CSV (the
      denominator for the volume percentage).

This script depends only on the stdlib. It does not import from ``models/``,
``Infrastructure/``, ``common/``, ``api/``, ``baselines/``, or any other
project module.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Truncation cap for the comma-joined regions string in the top-10 table.
# Keeps the table from blowing out terminal width on runs with many countries.
REGIONS_DISPLAY_MAX = 60


@dataclass
class CountryStats:
    """Running counts for a single country."""

    country: str
    total_measurements: int = 0
    blocked_domains: set[str] = field(default_factory=set)


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
    stats: CountryStats,
    domain_map: dict[str, dict[str, int]],
) -> None:
    """Stream ``csv_path`` once, updating ``stats`` and ``domain_map``.

    ``domain_map`` is keyed ``domain -> {country -> block_event_count}`` and
    is mutated in place for cross-region aggregation.
    """
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats.total_measurements += 1
            if row.get("is_blocked") != "1":
                continue
            target = (row.get("targets") or "").strip()
            if not target:
                # Empty target on a blocked row — still counts toward the
                # measurement total (already incremented above) but cannot
                # be tracked as a distinct domain.
                continue
            stats.blocked_domains.add(target)
            domain_map[target][country] += 1


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "\u2026"


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with fixed column widths (terminal-friendly)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def fmt(cells: list[str]) -> str:
        return (
            "| "
            + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))
            + " |"
        )

    lines = [fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    for row in rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def render_stats_block(domain_map: dict[str, dict[str, int]]) -> str:
    """Render the 3 single-line distinct-domain stats."""
    any_region = len(domain_map)
    exactly_one = sum(1 for regions in domain_map.values() if len(regions) == 1)
    three_plus = sum(1 for regions in domain_map.values() if len(regions) >= 3)
    return (
        f"Distinct domains blocked in any region: {any_region}\n"
        f"Distinct domains blocked in exactly one region: {exactly_one}\n"
        f"Distinct domains blocked in 3 or more regions: {three_plus}"
    )


def render_top_domains(domain_map: dict[str, dict[str, int]]) -> str:
    """Render the top-10 blocked-domain table.

    Sort key (DESC primary, ASC tiebreakers):
        (-regions_blocking, -total_block_events, domain ASC)

    Cross-region prevalence (``regions_blocking``) is the primary aggregate
    because raw event volume is dominated by measurement-volume skew between
    countries — a domain blocked in many regions is a more meaningful
    "global" signal than one blocked many times in a single high-volume
    region.
    """
    title = "### Top 10 blocked domains (aggregated across regions)"
    if not domain_map:
        return f"{title}\n_No blocked domains in this run._"

    enriched: list[tuple[str, int, int, list[str]]] = []
    for domain, regions in domain_map.items():
        sorted_countries = sorted(regions.keys())
        total_events = sum(regions.values())
        enriched.append((domain, len(sorted_countries), total_events, sorted_countries))

    enriched.sort(key=lambda t: (-t[1], -t[2], t[0]))
    top = enriched[:10]

    headers = ["rank", "domain", "regions_blocking", "total_block_events", "regions"]
    rows: list[list[str]] = []
    for idx, (domain, regions_blocking, total_events, countries) in enumerate(top, 1):
        regions_str = _truncate(", ".join(countries), REGIONS_DISPLAY_MAX)
        rows.append([
            str(idx),
            domain,
            str(regions_blocking),
            str(total_events),
            regions_str,
        ])
    return f"{title}\n{_render_markdown_table(headers, rows)}"


def render_per_country_sizes(country_stats: dict[str, CountryStats]) -> str:
    """Render the per-region blocked-domain-set-size table."""
    title = "### Blocked-domain set size per region"
    entries = [(c, len(s.blocked_domains)) for c, s in country_stats.items()]
    entries.sort(key=lambda t: (-t[1], t[0]))

    headers = ["country", "distinct_blocked_domains"]
    rows = [[country, str(count)] for country, count in entries]
    return f"{title}\n{_render_markdown_table(headers, rows)}"


def render_volume_table(country_stats: dict[str, CountryStats]) -> str:
    """Render the per-region measurement-volume table with pct_of_total."""
    title = "### Measurement volume per region"
    total = sum(s.total_measurements for s in country_stats.values())

    entries = [(c, s.total_measurements) for c, s in country_stats.items()]
    entries.sort(key=lambda t: (-t[1], t[0]))

    headers = ["country", "measurements", "pct_of_total"]
    rows: list[list[str]] = []
    for country, measurements in entries:
        if total > 0:
            pct_str = f"{(100 * measurements / total):.2f}%"
        else:
            pct_str = "0.00%"
        rows.append([country, str(measurements), pct_str])

    table = _render_markdown_table(headers, rows)
    if total == 0:
        return (
            f"{title}\n{table}\n"
            "_Note: zero total measurements; percentages are undefined._"
        )
    return f"{title}\n{table}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-region blocked-domain summary for a single CenRL run "
            "directory. Prints stats and 3 Markdown tables to stdout."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run directory (e.g. outputs/outtest19).",
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

    country_stats: dict[str, CountryStats] = {}
    domain_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for country, csv_path in country_csvs:
        stats = CountryStats(country=country)
        country_stats[country] = stats
        aggregate_country(country, csv_path, stats, domain_map)

    sections = [
        f"## Cross-region blocked-domain summary: {run_dir.name}",
        render_stats_block(domain_map),
        render_top_domains(domain_map),
        render_per_country_sizes(country_stats),
        render_volume_table(country_stats),
    ]
    print("\n\n".join(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
