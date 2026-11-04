"""
filter_false_positive_blocks.py

Standalone add-on -- does not touch orchestrator.py, node.py, or any
original run output. Re-derives the TRUE blocked/not-blocked verdict for
every "blocked" row in a run's data by cross-checking it against
Hyperquack's own raw per-VP `anomaly` field.

Why: orchestrator.py's _feed_aggregator() decides blocked per VP from
`response[0].matches_template` -- the FIRST connection attempt only. But
Hyperquack retries on a transient failure, and its own `anomaly` field
already accounts for the full retry sequence (and the control-domain
check). A VP whose first attempt got a one-off TCP reset but succeeded on
retry gets recorded "blocked" by CenRL even though Hyperquack's own,
retry-aware verdict says otherwise. With few enough VPs (3), one or two
such transient flips is enough to swing the majority vote.

This script recomputes each blocked target's verdict using the raw
anomaly votes instead, and only ever flips 1 -> 0 (removes a false
positive) -- never 0 -> 1. That asymmetry is intentional: Hyperquack does
not retry after a first-attempt SUCCESS, so a "not blocked" result was
never at risk of this bug to begin with; only "blocked" results need
re-checking.

Scope / what this does NOT do:
  - Only the is_blocked/blocked FLAG (and downstream aggregate stats) are
    corrected. rewards, q_value, and every RL-learning column are left
    untouched -- the model already learned from the buggy signal live
    during the run, so its actual exploration trajectory can't be
    retroactively un-learned by relabeling data after the fact. This is a
    reporting/analysis-level correction: "what does the true censorship
    picture look like," not "what would a bug-free run have produced."
  - A target with NO matching raw Hyperquack record for this run (e.g. it
    was re-routed to a different VP mid-run by the collision filter or a
    VP swap) is left unchanged and flagged unverified in the audit column
    -- rather than guessed at.

Usage:
    python3 filter_false_positive_blocks.py \\
        --run /home/cenrl/cenrl_outputs/initial_tests/outputs32 \\
        --hq-tests /home/cenrl/hyperquackv2/tests/23 \\
        --out-dir /home/cenrl/cenrl_outputs/initial_tests/outputs32_cleaned

The output directory mirrors the run's own <Country>/<Country>.csv and
<Country>/<Country>_measurements.csv layout, so analyze_censorship.py can
be pointed at it unmodified:
    python3 analyze_censorship.py --output_dir outputs32_cleaned --out outputs32_cleaned
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_country_raw_anomaly(hq_tests_dir: str, country: str) -> dict:
    """target -> (n_vps_reporting, n_raw_anomaly_true) from one country's raw
    Hyperquack *_result.jsonl. Multiple raw records for the same (target, vp)
    (resends) are deduped by keeping only the last one seen -- the final
    attempt for that VP on that target."""
    path = Path(hq_tests_dir) / f"{country}_result.jsonl"
    if not path.exists():
        return {}

    per_target_per_vp = defaultdict(dict)  # target -> {vp: anomaly}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            target = d.get("test_url")
            vp = d.get("vp")
            if target is None or vp is None:
                continue
            per_target_per_vp[target][vp] = d.get("anomaly", False)

    result = {}
    for target, vp_map in per_target_per_vp.items():
        n_vps = len(vp_map)
        n_anomaly = sum(1 for a in vp_map.values() if a)
        result[target] = (n_vps, n_anomaly)
    return result


def correct_dataframe(df: pd.DataFrame, blocked_col: str, target_col: str,
                       raw_anomaly: dict, force_exclude: bool = False) -> pd.DataFrame:
    """Returns a copy of df with blocked_col corrected, plus audit columns:
    raw_anomaly_votes, raw_n_vps, verified, flipped.

    force_exclude=True skips the raw-anomaly lookup entirely and zeroes out
    every blocked row for this country/file -- for a country whose VP pool
    is known to be bad (e.g. mislabeled-geolocation VPs persistently
    resetting every target, which the anomaly-field cross-check above
    can't catch since Hyperquack's own anomaly field agrees with those bad
    VPs). Coarser than the per-row anomaly check, but appropriate once a
    whole country's pool is known contaminated rather than one-off noise."""
    df = df.copy()
    raw_votes = []
    raw_n = []
    verified = []
    flipped = []

    for _, row in df.iterrows():
        was_blocked = bool(row[blocked_col])
        target = row[target_col]
        if not was_blocked:
            # Never at risk of this bug -- leave untouched.
            raw_votes.append(None)
            raw_n.append(None)
            verified.append(True)
            flipped.append(False)
            continue

        if force_exclude:
            raw_votes.append(None)
            raw_n.append(None)
            verified.append(True)
            flipped.append(True)
            continue

        lookup = raw_anomaly.get(target)
        if lookup is None:
            raw_votes.append(None)
            raw_n.append(None)
            verified.append(False)
            flipped.append(False)
            continue

        n_vps, n_anomaly = lookup
        would_still_be_blocked = n_anomaly > n_vps / 2
        raw_votes.append(n_anomaly)
        raw_n.append(n_vps)
        verified.append(True)
        flipped.append(not would_still_be_blocked)

    df["raw_anomaly_votes"] = raw_votes
    df["raw_n_vps"] = raw_n
    df["verified"] = verified
    df["flipped_false_positive"] = flipped

    df.loc[df["flipped_false_positive"], blocked_col] = 0
    return df


def process_run(run_dir: str, hq_tests_dir: str, out_dir: str, exclude_countries: list = None):
    exclude_countries = set(exclude_countries or [])
    run_path = Path(run_dir)
    country_dirs = [p for p in run_path.iterdir() if p.is_dir() and p.name != "state"]

    total_blocked = 0
    total_flipped = 0
    total_unverified = 0

    for country_dir in sorted(country_dirs):
        country_dir_name = country_dir.name
        country = country_dir_name.replace("_", " ")
        force_exclude = country in exclude_countries

        raw_anomaly = load_country_raw_anomaly(hq_tests_dir, country)

        out_country_dir = Path(out_dir) / country_dir_name
        out_country_dir.mkdir(parents=True, exist_ok=True)

        # Rolling <Country>.csv (used by load_all() for most plots)
        rolling_csv = country_dir / f"{country_dir_name}.csv"
        if rolling_csv.exists():
            df = pd.read_csv(rolling_csv)
            if "is_blocked" in df.columns and "targets" in df.columns:
                before = df["is_blocked"].sum()
                df = correct_dataframe(df, "is_blocked", "targets", raw_anomaly, force_exclude=force_exclude)
                after = df["is_blocked"].sum()
                n_flipped = int(before - after)
                n_unverified = int((~df["verified"]).sum())
                total_blocked += int(before)
                total_flipped += n_flipped
                total_unverified += n_unverified
                tag = " [COUNTRY EXCLUDED -- known-bad VP pool]" if force_exclude else ""
                print(f"{country:18s} rolling CSV: {int(before):4d} blocked -> {int(after):4d} "
                      f"({n_flipped} false positives removed, {n_unverified} unverified){tag}")
                df.to_csv(out_country_dir / f"{country_dir_name}.csv", index=False)

        # <Country>_measurements.csv (used by blocked_domains_with_confidence())
        meas_csv = country_dir / f"{country_dir_name}_measurements.csv"
        if meas_csv.exists():
            mdf = pd.read_csv(meas_csv)
            if "blocked" in mdf.columns and "target" in mdf.columns:
                mdf = correct_dataframe(mdf, "blocked", "target", raw_anomaly, force_exclude=force_exclude)
                mdf.to_csv(out_country_dir / f"{country_dir_name}_measurements.csv", index=False)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_blocked} originally-blocked entries, "
          f"{total_flipped} false positives removed ({total_flipped/total_blocked:.1%}), "
          f"{total_unverified} unverified (left unchanged, no matching raw record)")
    print(f"Cleaned data written to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove first-attempt-only false positives from a run's blocked results, "
                     "using Hyperquack's own raw anomaly field as ground truth."
    )
    parser.add_argument("--run", required=True, help="Run output directory, e.g. outputs32")
    parser.add_argument("--hq-tests", required=True,
                        help="Matching Hyperquack tests/N directory with the raw *_result.jsonl files")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write the corrected copy into (mirrors --run's layout)")
    parser.add_argument("--exclude-countries", nargs="+", default=[],
                        help='Countries whose entire blocked results should be zeroed out '
                             '(known-bad VP pool, e.g. --exclude-countries "United Kingdom") '
                             'rather than checked row-by-row against the raw anomaly field')
    args = parser.parse_args()

    process_run(args.run, args.hq_tests, args.out_dir, exclude_countries=args.exclude_countries)


if __name__ == "__main__":
    main()
