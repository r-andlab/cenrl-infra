"""
Vantage Point Statistics

Reads ev-certs.csv and the blocklist to produce stats on the vantage point
pool: totals, blocklist hits, per-country breakdowns.  Non-blocked VPs are
exported to a CSV keyed by country for later use.

Usage:
    python -m Infrastructure.main.vantage_point_stats \
        --ev-file ev-certs.csv \
        --blocklist blocklist.txt \
        --stats-out vp_stats.txt \
        --csv-out vp_pool.csv
"""

import argparse
import csv
import ipaddress
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


def load_blocklist(path: str) -> List[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    networks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            networks.append(ipaddress.ip_network(line, strict=False))
    return networks


def is_blocked(ip_str: str, blocklist: list) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # treat unparseable IPs as blocked
    return any(addr in net for net in blocklist)


def gather_stats(ev_file: str, blocklist_file: str):
    df = pd.read_csv(ev_file)
    blocklist = load_blocklist(blocklist_file)

    total_rows = len(df)
    total_no_ip = 0
    total_blocked = 0
    total_valid = 0

    # country -> {"valid": set((ip, port)), "blocked": set((ip, port))}
    per_country: Dict[str, Dict[str, Set[Tuple[str, str]]]] = defaultdict(
        lambda: {"valid": set(), "blocked": set()}
    )

    for row in df.itertuples(index=False):
        ip = None
        if hasattr(row, "ipv4") and row.ipv4 and not pd.isna(row.ipv4):
            ip = str(row.ipv4).strip()
        elif hasattr(row, "ipv6") and row.ipv6 and not pd.isna(row.ipv6):
            ip = str(row.ipv6).strip()

        if not ip:
            total_no_ip += 1
            continue

        port = str(int(row.port)) if hasattr(row, "port") and not pd.isna(row.port) else ""

        country = getattr(row, "country", None)
        if not country or pd.isna(country):
            country = "Unknown"
        country = str(country).strip()

        if is_blocked(ip, blocklist):
            total_blocked += 1
            per_country[country]["blocked"].add((ip, port))
        else:
            total_valid += 1
            per_country[country]["valid"].add((ip, port))

    return total_rows, total_no_ip, total_blocked, total_valid, per_country


def format_report(total_rows, total_no_ip, total_blocked, total_valid, per_country):
    lines = []
    lines.append("=" * 60)
    lines.append("VANTAGE POINT STATISTICS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total rows in ev-certs.csv:    {total_rows}")
    lines.append(f"  Rows without a usable IP:    {total_no_ip}")
    lines.append(f"  IPs on blocklist:            {total_blocked}")
    lines.append(f"  Valid (non-blocked) IPs:      {total_valid}")
    lines.append("")

    countries_sorted = sorted(per_country.keys())
    lines.append(f"Total countries:               {len(countries_sorted)}")
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"{'Country':<30} {'Valid':>8} {'Blocked':>8} {'Total':>8}")
    lines.append("-" * 60)

    for country in countries_sorted:
        entry = per_country[country]
        n_valid = len(entry["valid"])
        n_blocked = len(entry["blocked"])
        lines.append(
            f"{country:<30} {n_valid:>8} {n_blocked:>8} {n_valid + n_blocked:>8}"
        )

    lines.append("-" * 60)
    lines.append("")
    return "\n".join(lines)


def write_vp_csv(path: str, per_country):
    """Write all valid (non-blocked) VPs to a CSV: country, ip, port."""
    countries_sorted = sorted(per_country.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "ip", "port"])
        for country in countries_sorted:
            for ip, port in sorted(per_country[country]["valid"]):
                writer.writerow([country, ip, port])


def main():
    parser = argparse.ArgumentParser(description="Vantage point pool statistics")
    parser.add_argument(
        "--ev-file", default="ev-certs.csv", help="Path to ev-certs CSV"
    )
    parser.add_argument(
        "--blocklist", default="blocklist.txt", help="Path to blocklist file"
    )
    parser.add_argument(
        "--stats-out",
        default="vp_stats.txt",
        help="File to write the stats report to",
    )
    parser.add_argument(
        "--csv-out",
        default="vp_pool.csv",
        help="CSV output of valid VPs (country, ip)",
    )
    args = parser.parse_args()

    total_rows, total_no_ip, total_blocked, total_valid, per_country = gather_stats(
        args.ev_file, args.blocklist
    )

    report = format_report(
        total_rows, total_no_ip, total_blocked, total_valid, per_country
    )

    print(report)

    with open(args.stats_out, "w") as f:
        f.write(report)
    print(f"Stats written to {args.stats_out}")

    write_vp_csv(args.csv_out, per_country)
    print(f"Valid VPs written to {args.csv_out}")


if __name__ == "__main__":
    main()
