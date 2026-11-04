"""
build_asn_matched_priors.py

Extends build_warm_start_priors.py with an ASN-specific sibling prior file
per (country, ASN): <out-dir>/<Country>/asn_<ASN>.csv, in the identical
format write_priors_csv() already produces (reused here, not reimplemented).
orchestrator.py picks whichever file matches the country's actually-
confirmed VP's resolved ASN at node-creation time, falling back to the
existing country-wide file (still built by build_warm_start_priors.py,
unchanged) when no ASN-specific data exists yet for that ASN.

Why: the country-wide prior blends every past VP's history into one number,
so how biased it is for THIS run depends entirely on which VP this run
happens to draw -- which is exactly the mechanism behind the hard-50/
hard-100 degradation (see the warm-start N comparison writeup). ASNs,
unlike individual IPs, persist strongly across the monthly VP pool refresh
(~85-99% by volume between the April and September pulls, verified
empirically) -- so a per-ASN prior is a much better-targeted estimate for
whichever specific VP this run actually lands on. This shrinks the prior's
bias without changing the live bandit's arm space at all: arms are still
just categories, exactly as before.

What it does, per (run, hq-tests) pair, per country:
  1. Reads the run's own <Country>_measurements.csv for a target -> category
     (arm) lookup -- built by the run itself, so it's always in sync with
     that run's own category taxonomy.
  2. Reads the raw Hyperquack <Country>_result.jsonl (the same file
     filter_false_positive_blocks.py already uses) for (vp, test_url,
     anomaly) -- Hyperquack's own retry-aware per-VP verdict, corrected for
     Bug #1 at the source rather than via the country-level majority vote.
  3. Resolves each vp IP's ASN via the ev-certs pool CSV's own `asn` column
     -- the same file every listed run's VP pool was drawn from (spot-
     checked to agree with external WHOIS/RDAP, e.g. 198.97.13.235 -> ASN
     62854, Cheney Bros -- see the Cheney Bros/Verizon false-positive
     investigation this same pipeline caught).
  4. Accumulates anomaly observations into (asn, category) buckets, pooled
     across every given run.
  5. Writes one prior file per ASN with >= --min-observations pooled real
     observations for at least one category, via write_priors_csv() --
     byte-for-byte the same format as the country-wide file, just scoped to
     one ASN.

Usage:
    python3 build_asn_matched_priors.py \\
        --runs /path/to/outputs35_cleaned /path/to/outputs38_cleaned \\
        --hq-tests /path/to/tests/26 /path/to/tests/29 \\
        --countries China "South Korea" \\
        --ev-certs /path/to/ev-certs-monthly-000000000000.csv.gz \\
        --out-dir /path/to/priors \\
        --fake-attempts 5
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from Infrastructure.main.build_warm_start_priors import write_priors_csv


def load_ip_asn_map(ev_certs_path: str) -> dict:
    """ipv4 -> asn (as str), straight from the pool CSV every listed run
    drew its VPs from. Rows with no ASN are dropped rather than guessed."""
    df = pd.read_csv(ev_certs_path)
    df = df.dropna(subset=["asn"])
    return dict(zip(df["ipv4"], df["asn"].astype(int).astype(str)))


def target_category_map(measurements_csv: str) -> dict:
    df = pd.read_csv(measurements_csv)
    return dict(zip(df["target"], df["arm"]))


def accumulate_asn_observations(run_dirs: list, hq_tests_dirs: list, country: str,
                                 ip_asn: dict) -> dict:
    """asn -> category -> [anomaly (0.0/1.0), ...], pooled across every
    given run. Uses each run's own raw per-VP anomaly field directly (not
    the country-level majority vote), so this is Bug-#1-clean by
    construction -- the retry-aware verdict IS the observation."""
    country_dir_name = country.replace(" ", "_")
    obs: dict = defaultdict(lambda: defaultdict(list))

    for run_dir, hq_dir in zip(run_dirs, hq_tests_dirs):
        meas_path = Path(run_dir) / country_dir_name / f"{country_dir_name}_measurements.csv"
        raw_path = Path(hq_dir) / f"{country}_result.jsonl"
        if not meas_path.exists() or not raw_path.exists():
            missing = meas_path.name if not meas_path.exists() else raw_path.name
            print(f"  skip {run_dir}: missing {missing}")
            continue

        cat_map = target_category_map(str(meas_path))
        n_used = 0
        with open(raw_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vp = d.get("vp")
                target = d.get("test_url")
                if vp is None or target is None:
                    continue
                category = cat_map.get(target)
                if category is None:
                    continue
                asn = ip_asn.get(vp)
                if asn is None:
                    continue
                obs[asn][category].append(1.0 if d.get("anomaly") else 0.0)
                n_used += 1
        print(f"  {run_dir}: {n_used} raw observations resolved to a known ASN for {country}")

    return obs


def write_asn_priors(obs: dict, country: str, out_dir: str, fake_attempts: int,
                      min_observations: int) -> int:
    n_written = 0
    for asn, cats in sorted(obs.items()):
        averaged = {
            cat: sum(vals) / len(vals)
            for cat, vals in cats.items()
            if len(vals) >= min_observations
        }
        if not averaged:
            continue
        total_obs = sum(len(v) for v in cats.values())
        out_path = write_priors_csv(averaged, country, out_dir, fake_attempts=fake_attempts,
                                     filename=f"asn_{asn}.csv")
        print(f"  ASN {asn:>8s}: {len(averaged)} arm priors ({total_obs} pooled observations) -> {out_path}")
        n_written += 1
    return n_written


def main():
    parser = argparse.ArgumentParser(
        description="Build per-(country, ASN) warm-start priors from past runs' raw per-VP data."
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Output directories of past (cleaned) runs, e.g. outputs35_cleaned")
    parser.add_argument("--hq-tests", nargs="+", required=True,
                        help="Matching Hyperquack tests/N dirs, same order/length as --runs")
    parser.add_argument("--countries", nargs="+", required=True,
                        help='Countries to build ASN priors for, e.g. China "South Korea"')
    parser.add_argument("--ev-certs", required=True,
                        help="The pool CSV every listed run drew its VPs from (for ip -> asn)")
    parser.add_argument("--out-dir", required=True,
                        help="Same --out-dir as the matching build_warm_start_priors.py run -- "
                             "these files land alongside the country-wide fallback, not replace it")
    parser.add_argument("--fake-attempts", type=int, default=0)
    parser.add_argument("--min-observations", type=int, default=3,
                        help="Minimum pooled real observations for an (asn, category) to get a prior")
    args = parser.parse_args()

    if len(args.runs) != len(args.hq_tests):
        parser.error("--runs and --hq-tests must be given in the same order and count")

    ip_asn = load_ip_asn_map(args.ev_certs)
    print(f"Loaded {len(ip_asn)} ip->asn mappings from {args.ev_certs}")
    os.makedirs(args.out_dir, exist_ok=True)

    for country in args.countries:
        print(f"\n=== {country} ===")
        obs = accumulate_asn_observations(args.runs, args.hq_tests, country, ip_asn)
        n = write_asn_priors(obs, country, args.out_dir, args.fake_attempts, args.min_observations)
        print(f"  {n} ASN-specific prior file(s) written for {country}")


if __name__ == "__main__":
    main()
