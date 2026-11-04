r"""
export_asn_votes.py

Export per-(country, target, ASN) vote breakdowns from an all-VPs run's raw
Hyperquack results (<run_dir>/<Country>_result.jsonl or <run_dir>/hq/
<Country>_result.jsonl, one record per VP x target), for building ASN-level
heatmaps/analysis -- e.g. comparing a flat per-VP block rate against an
ASN-grouped one for the same raw votes.

IP -> ASN mapping comes from AsnAwareVantagePoints.get_asn() (Infrastructure/
main/asn_aware_vantage_points.py), which resolves each VP's ASN via CAIDA's
pfx2as database (local/ipasn_20260720.dat) first, falling back to the
ev-certs census CSV's own ASN column, and finally to the sentinel "unknown"
if neither resolves. Same resolver the real run uses for asn_vote_fraction
(Infrastructure/utils/aggregator.py).

For each (country, target), this groups every VP's raw per-probe "anomaly"
verdict -- the same retry-aware blocked signal Orchestrator._feed_aggregator
actually votes with (`blocked = r.anomaly`; see the Bug #1 fix comment there
-- NOT the older first-attempt-only `stateful_block or first_mismatch`
formula aggregate_all_vps_results.py's "vote" field predates) -- by ASN, and
writes one row per (country, target, asn) with how many VPs in that ASN
voted, how many voted blocked, and the resulting within-ASN vote fraction. A
VP whose ASN can't be resolved gets its own singleton "vp:<ip>" group, same
convention as aggregator.py's asn_vote_fraction, rather than being merged
with other unresolvable VPs under one artificial "unknown" bucket.

Each row also carries the category the bandit actually attributed to that
target for this run, read from <run_dir>/<Country>/<Country>_measurements.csv's
`arm` column (not the master tranco action-space file, which can list
multiple categories per domain -- the run's own measurements.csv reflects
which single category CenRL picked for it this run, same mapping used in
earlier ad hoc analysis this session). A target with no matching
measurements.csv row (e.g. it never finalized) gets category "unknown".

Usage (from /home/cenrl/cenrl):
    python3 -m Infrastructure.main.export_asn_votes \
        --run-dir ~/cenrl_outputs/experiment_six/output12 \
        --out ~/asn_votes_exp6.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from Infrastructure.main.asn_aware_vantage_points import AsnAwareVantagePoints, _UNKNOWN_ASN

DEFAULT_COUNTRIES = ["China", "Germany", "Iran", "Thailand", "Turkey",
                     "United Arab Emirates", "Vietnam"]


def find_result_file(run_dir: str, country: str) -> str | None:
    for candidate in (
        os.path.join(run_dir, "hq", f"{country}_result.jsonl"),
        os.path.join(run_dir, f"{country}_result.jsonl"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def find_measurements_file(run_dir: str, country: str) -> str | None:
    country_us = country.replace(" ", "_")
    for candidate in (
        os.path.join(run_dir, country, f"{country}_measurements.csv"),
        os.path.join(run_dir, "app", country, f"{country}_measurements.csv"),
        os.path.join(run_dir, country_us, f"{country_us}_measurements.csv"),
        os.path.join(run_dir, "app", country_us, f"{country_us}_measurements.csv"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def load_target_categories(run_dir: str, country: str) -> dict:
    """target -> category (the 'categories <Name>' arm stripped down to
    just <Name>), from this run's own measurements.csv -- the single
    category the bandit actually attributed to that target this run."""
    path = find_measurements_file(run_dir, country)
    if path is None:
        return {}
    cats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            arm = row.get("arm", "")
            cats[row["target"]] = arm[len("categories "):] if arm.startswith("categories ") else arm
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory containing <Country>_result.jsonl")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    ap.add_argument("--ev-file", default="local/ev-certs-monthly-202610-000000000000.csv.gz")
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    out_csv = os.path.expanduser(args.out)

    pool = AsnAwareVantagePoints(
        ev_file=args.ev_file,
        blocklist_file="local/blocklist.txt",
        max_countries=0,
        blocked_countries=[],
        caida_dat="local/ipasn_20260720.dat",
        guaranteed_countries=args.countries,
        excluded_asns=["714", "6185", "31898"],
        as_org_path=None,
        excluded_vps_file="local/excluded_vps.txt",
    )

    asn_cache = {}

    def get_asn_key(vp: str) -> str:
        key = asn_cache.get(vp)
        if key is None:
            asn = pool.get_asn(vp)
            key = f"vp:{vp}" if asn == _UNKNOWN_ASN else f"asn:{asn}"
            asn_cache[vp] = key
        return key

    with open(out_csv, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["country", "target", "category", "asn", "n_votes", "n_blocked", "vote_fraction"])

        for country in args.countries:
            path = find_result_file(run_dir, country)
            if path is None:
                print(f"{country}: no result file found, skipping")
                continue

            target_categories = load_target_categories(run_dir, country)
            if not target_categories:
                print(f"{country}: no measurements.csv found, category will be 'unknown' for every row")

            # (target, asn_key) -> [n_blocked, n_total]
            counts = defaultdict(lambda: [0, 0])
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    asn_key = get_asn_key(r["vp"])
                    c = counts[(r["test_url"], asn_key)]
                    c[1] += 1
                    if r["anomaly"]:
                        c[0] += 1

            for (target, asn_key), (n_blocked, n_total) in counts.items():
                category = target_categories.get(target, "unknown")
                writer.writerow([country, target, category, asn_key, n_total, n_blocked, round(n_blocked / n_total, 4)])

            print(f"{country}: wrote {len(counts)} (target, asn) rows")

    print(f"Done. Output: {out_csv}")


if __name__ == "__main__":
    main()
