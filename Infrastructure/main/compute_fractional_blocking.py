r"""
compute_fractional_blocking.py

Compute a flat, per-target "fractional blocking" score (0-1) from raw
Hyperquack per-probe results -- the same vote_fraction formula aggregator.py
uses (sum of per-VP anomaly votes over the number of voting VPs, no ASN
grouping), for a run that doesn't already have it: the No-CenRL baseline
(flat is_blocked only, no reward concept at all) and binary/majority-reward
CenRL runs (reward == is_blocked, 0/1 only). Runs that already used
--reward-mode percentage or asn-percentage already have a usable fractional
column in their own measurements.csv and don't need this.

Takes raw result files directly rather than assuming a directory layout,
since the baseline's raw output is split across several timestamped
CP_Quack-*/output.json files per country while CenRL-driven runs write one
<Country>_result.jsonl -- both are newline-delimited JSON with the same
{vp, test_url, anomaly, ...} schema, so globbing either kind in is fine.

Category comes from an existing measurements.csv's target -> arm mapping
(same convention used throughout this project: the first category listed
for a domain), not recomputed here.

Usage (from /home/cenrl/cenrl):
    python3 -m Infrastructure.main.compute_fractional_blocking \
        --country China \
        --result-files '/home/cenrl/cenrl_outputs/hyperquack_control_census/runs/china/results/https/CP_Quack-*/output.json' \
        --measurements-csv /home/cenrl/cenrl_outputs/hyperquack_control_census/analysis_input/China/China_measurements.csv \
        --out /home/cenrl/cenrl_outputs/hyperquack_control_census/analysis_input/China/China_fractional.csv
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def load_target_categories(measurements_csv: str) -> dict:
    cats = {}
    with open(os.path.expanduser(measurements_csv), newline="") as f:
        for row in csv.DictReader(f):
            arm = row.get("arm", "")
            cats[row["target"]] = arm[len("categories "):] if arm.startswith("categories ") else arm
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--result-files", required=True,
                     help="Glob pattern (quoted) matching one or more raw *_result.jsonl / output.json files")
    ap.add_argument("--measurements-csv", required=True,
                     help="Existing measurements.csv to source target -> category from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target_categories = load_target_categories(args.measurements_csv)

    files = sorted(glob.glob(os.path.expanduser(args.result_files)))
    if not files:
        raise SystemExit(f"No files matched: {args.result_files}")
    print(f"{args.country}: reading {len(files)} file(s)")

    # target -> [n_blocked, n_total]
    counts = defaultdict(lambda: [0, 0])
    bad = 0
    for path in files:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                c = counts[r["test_url"]]
                c[1] += 1
                if r["anomaly"]:
                    c[0] += 1
    if bad:
        print(f"{args.country}: skipped {bad} unparseable lines")

    out_path = os.path.expanduser(args.out)
    with open(out_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["country", "target", "category", "n_votes", "n_blocked", "vote_fraction"])
        for target, (n_blocked, n_total) in counts.items():
            category = target_categories.get(target, "unknown")
            writer.writerow([args.country, target, category, n_total, n_blocked, round(n_blocked / n_total, 4)])

    print(f"{args.country}: wrote {len(counts)} target rows -> {out_path}")


if __name__ == "__main__":
    main()
