"""Detect VP/target CDN-ASN collision false positives in Hyperquack results.

Per Sundara Raman et al., "Advancing the Art of Censorship Data Analysis"
(FOCI 2023), Hyperquack sends an HTTP request for a target domain's Host
header to an unrelated vantage-point IP, expecting an error response. If the
VP and the target domain happen to be hosted on the *same* CDN/ASN, the CDN's
edge network may route by Host header anyway and return the target's real
(non-error) response instead -- producing an "anomalous"/blocked-looking
result that has nothing to do with censorship.

This script reads raw per-VP results directly from a Hyperquack --logoutdir
(<hq-dir>/<Country>_result.jsonl), since VP identity does not exist in the
CenRL orchestrator's own per-country CSVs. For each anomalous (candidate-
blocked) result it looks up the VP's ASN and the target domain's ASN (via a
fresh DNS resolution + the same CAIDA pfx2as database) and flags same-ASN
collisions as likely false positives rather than genuine blocking.

Requires: pip install pyasn, plus a CAIDA pfx2as .dat file built via
Infrastructure/main/convert_caida.py.

Produces, under <hq-dir>/plots/ (or --out):
  - cdn_collision_summary.png      per-country counts: collision vs non-collision anomalies
  - cdn_collision_domains.png      text table of (country, vp, domain, shared ASN)
  - cdn_collision_analysis.csv     full raw joined table for inspection

Example:
    python3 Infrastructure/scripts/plotting/analyze_cdn_collisions.py \\
        --hq-dir /home/cenrl/hyperquackv2/tests/11 \\
        --asn-db local/ipasn_20260720.dat
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import pyasn
import seaborn as sns

from Infrastructure.scripts.plotting._style import (
    apply_paper_style,
    ensure_out_dir,
    savefig,
)

DNS_TIMEOUT_S = 3.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flag VP/target CDN-ASN collisions in raw Hyperquack results."
    )
    p.add_argument(
        "--hq-dir",
        required=True,
        type=Path,
        help="Hyperquack --logoutdir directory (contains <Country>_result.jsonl files).",
    )
    p.add_argument(
        "--asn-db",
        required=True,
        type=Path,
        help="pyasn .dat file built via Infrastructure/main/convert_caida.py.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output directory. Default: <hq-dir>/plots/",
    )
    return p.parse_args()


def _is_anomalous(row: dict) -> bool:
    """Same blocked-heuristic used elsewhere in the pipeline (orchestrator.py
    _feed_aggregator, analyze_censorship.py): stateful_block, or the first
    response not matching the VP's established template."""
    if row.get("stateful_block"):
        return True
    response = row.get("response") or []
    if response and not response[0].get("matches_template", True):
        return True
    return False


def _resolve_domain_ip(domain: str, cache: dict) -> Optional[str]:
    if domain in cache:
        return cache[domain]
    try:
        ip = socket.gethostbyname(domain)
    except Exception:
        ip = None
    cache[domain] = ip
    return ip


def _lookup_asn(ip: Optional[str], asndb: "pyasn.pyasn", cache: dict) -> Optional[int]:
    if ip is None:
        return None
    if ip in cache:
        return cache[ip]
    try:
        asn, _prefix = asndb.lookup(ip)
    except Exception:
        asn = None
    cache[ip] = asn
    return asn


def load_results(hq_dir: Path, asndb: "pyasn.pyasn") -> pd.DataFrame:
    """Parse every <Country>_result.jsonl in hq_dir into one long DataFrame:
    country, vp, vp_asn, test_url, domain_asn, is_anomalous, same_asn_collision.
    """
    socket.setdefaulttimeout(DNS_TIMEOUT_S)
    vp_asn_cache: dict = {}
    domain_ip_cache: dict = {}
    domain_asn_cache: dict = {}

    rows = []
    result_files = sorted(hq_dir.glob("*_result.jsonl"))
    for path in result_files:
        country = path.stem[: -len("_result")].replace("_", " ")
        n_lines = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_lines += 1
                vp = rec.get("vp")
                domain = rec.get("test_url")
                if not vp or not domain:
                    continue

                vp_asn = _lookup_asn(vp, asndb, vp_asn_cache)
                domain_ip = _resolve_domain_ip(domain, domain_ip_cache)
                domain_asn = _lookup_asn(domain_ip, asndb, domain_asn_cache)
                anomalous = _is_anomalous(rec)
                same_asn = (
                    vp_asn is not None
                    and domain_asn is not None
                    and vp_asn == domain_asn
                )
                rows.append(
                    {
                        "country": country,
                        "vp": vp,
                        "vp_asn": vp_asn,
                        "test_url": domain,
                        "domain_ip": domain_ip,
                        "domain_asn": domain_asn,
                        "is_anomalous": anomalous,
                        "same_asn_collision": anomalous and same_asn,
                    }
                )
        print(f"  parsed {n_lines} results from {path.name}")

    if not rows:
        raise FileNotFoundError(f"No *_result.jsonl files found under {hq_dir}")
    return pd.DataFrame(rows)


def plot_collision_summary(df: pd.DataFrame, out_dir: Path) -> None:
    anomalous = df[df["is_anomalous"]]
    if anomalous.empty:
        print("  no anomalous results at all; skipping cdn_collision_summary.png")
        return

    summary = (
        anomalous.groupby("country")["same_asn_collision"]
        .agg(collisions="sum", total="count")
        .assign(genuine=lambda d: d["total"] - d["collisions"])
        .sort_values("total", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    countries = summary.index.tolist()
    ax.barh(countries, summary["collisions"], color=sns.color_palette("flare")[0],
            label="Same-ASN collision (likely false positive)")
    ax.barh(countries, summary["genuine"], left=summary["collisions"],
            color=sns.color_palette("crest")[3], label="Different ASN (likely genuine)")
    ax.set_xlabel("Anomalous (candidate-blocked) results")
    ax.set_title("VP/Target CDN-ASN Collisions Among Anomalous Results, by Country",
                  fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    sns.despine(fig=fig)
    fig.tight_layout()
    savefig(fig, out_dir, "cdn_collision_summary")


def plot_collision_domains(df: pd.DataFrame, out_dir: Path) -> None:
    collisions = df[df["same_asn_collision"]].drop_duplicates(
        subset=["country", "test_url", "vp"]
    )
    if collisions.empty:
        print("  no same-ASN collisions found; skipping cdn_collision_domains.png")
        return

    rows = []
    for _, r in collisions.sort_values(["country", "test_url"]).iterrows():
        rows.append(
            f"  {r['country']:<16} {r['test_url']:<32} "
            f"vp={r['vp']:<16} ASN {r['domain_asn']} (shared)"
        )

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.24)))
    ax.axis("off")
    text = "Likely CDN-Collision False Positives (VP ASN == Target Domain ASN)\n\n"
    text += f"  {'Country':<16} {'Domain':<32} {'Vantage Point'}\n"
    text += "  " + "-" * 70 + "\n"
    text += "\n".join(rows)
    ax.text(0.01, 0.99, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", family="monospace")
    fig.tight_layout()
    savefig(fig, out_dir, "cdn_collision_domains")


def main() -> int:
    apply_paper_style()
    args = _parse_args()

    hq_dir: Path = args.hq_dir
    out_dir = ensure_out_dir(hq_dir, args.out)

    print(f"Loading ASN database from {args.asn_db}")
    asndb = pyasn.pyasn(str(args.asn_db))

    print(f"Parsing raw Hyperquack results from {hq_dir}")
    df = load_results(hq_dir, asndb)

    csv_path = out_dir / "cdn_collision_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    n_anomalous = int(df["is_anomalous"].sum())
    n_collision = int(df["same_asn_collision"].sum())
    print(
        f"\n{len(df):,} total results, {n_anomalous:,} anomalous "
        f"(candidate-blocked), {n_collision:,} of those are same-ASN "
        f"CDN collisions ({n_collision / n_anomalous * 100:.1f}% of anomalies)"
        if n_anomalous else f"\n{len(df):,} total results, 0 anomalous"
    )

    figure_fns = (
        ("summary", plot_collision_summary),
        ("domains", plot_collision_domains),
    )
    for name, fn in figure_fns:
        try:
            fn(df, out_dir)
        except Exception as exc:
            print(f"  ERROR plotting {name}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
