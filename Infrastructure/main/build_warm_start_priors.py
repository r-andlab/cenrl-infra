"""
build_warm_start_priors.py

Standalone add-on script -- does NOT touch any existing model/action-space
code. Builds one priors CSV per country, in the exact shape
`read_action_value_file()` already expects (episode,time,action,q_value),
laid out as `<out-dir>/<Country_Name>/<Country_Name>.csv` -- exactly the
folder structure `RegionalNode.__init__` already scans for via its
`action_space_folder` argument (node.py:45-52). Point a run's
`--previous-values-folder` at this script's --out-dir and each country
picks up ONLY its own file automatically; a country with no matching
file/folder here just falls through to a normal cold start untouched.
That per-country isolation is the whole reason for this layout -- see the
orchestrator.py change alongside this script for the other half (exposing
--previous-values-folder on the CLI, since that constructor param existed
but was never wired up before).

What it does:
  1. For each (run, country) pair, replicates read_action_value_file()'s own
     extraction logic (models/base/action_space.py:146-159) independently:
     sort by (episode, time), take the LAST q_value per (episode, action),
     then mean across episodes -- giving one final learned value per arm for
     that single run.
  2. Averages that per-arm value ACROSS the given runs, per country kept
     SEPARATE (a run's China value is only ever averaged with another run's
     China value, never blended with South Korea's).
  3. Writes each country's averaged values to its own file/folder. Countries
     are never merged into one file, since a merged file would defeat the
     per-country isolation the folder layout is built to provide.

Usage:
    python3 build_warm_start_priors.py \\
        --runs /path/to/outputs27 /path/to/outputs28 \\
        --countries China "South Korea" \\
        --out-dir /path/to/priors
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def extract_final_arm_values(csv_path: str) -> dict:
    """One run's final learned q_value per arm.

    Mirrors read_action_value_file() (models/base/action_space.py:146-159)
    exactly: last q_value per (episode, action), then mean across episodes.
    Kept as a small standalone re-implementation rather than importing
    ActionSpaceBase, since that class also wants a full action-space
    dataframe/target_feature/etc. to construct -- overkill for extracting
    a dict from an already-written CSV.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["episode", "time"])
    last_occurrences = df.groupby(["episode", "action"]).last().reset_index()
    action_averages = (
        last_occurrences.groupby("action")[["q_value"]].mean().reset_index()
    )
    return action_averages.set_index("action")["q_value"].to_dict()


def average_across_runs(run_dirs: list, country: str) -> dict:
    """Per-arm mean of extract_final_arm_values() across every run that has
    data for this country. An arm present in only some runs is averaged over
    just those (not treated as 0 for the runs missing it)."""
    country_dir_name = country.replace(" ", "_")
    per_run_values = []
    for run_dir in run_dirs:
        csv_path = Path(run_dir) / country_dir_name / f"{country_dir_name}.csv"
        if not csv_path.exists():
            print(f"  skip {run_dir}: no {csv_path.name} found for {country}")
            continue
        values = extract_final_arm_values(str(csv_path))
        print(f"  {run_dir}: {len(values)} arms extracted for {country}")
        per_run_values.append(values)

    if not per_run_values:
        return {}

    all_arms = set()
    for values in per_run_values:
        all_arms.update(values.keys())

    averaged = {}
    for arm in all_arms:
        observed = [v[arm] for v in per_run_values if arm in v]
        averaged[arm] = sum(observed) / len(observed)
    return averaged


def write_priors_csv(averaged: dict, country: str, out_dir: str, fake_attempts: int = 0,
                      filename: str = None) -> str:
    """One row per arm, in the episode,time,action,q_value shape
    read_action_value_file() requires. episode/time are dummy constants
    (1, 1) -- there's exactly one row per (episode, action) group by
    construction here, so its own last-occurrence-then-mean logic passes
    each value through unchanged when this file is loaded back in.

    Written to <out_dir>/<Country_Name>/<Country_Name>.csv -- the exact
    layout RegionalNode.__init__ already looks for via action_space_folder
    (node.py:45-52): action_space_csv = path / country_name_standard /
    f"{country_name_standard}.csv". This is what makes the file apply ONLY
    to this one country -- a country whose folder/file isn't present here
    just falls through to a normal cold start, untouched.

    fake_attempts > 0 additionally writes an action_attempts column (hard
    warm-start, models/base/action_space.py's read_action_value_file()/
    build_graph() now read it) -- but ONLY for arms with q_value > 0. An
    arm that averaged to exactly 0.0 across the source runs has no real
    signal behind it (2 runs isn't enough evidence it's truly
    uninteresting), so it's left at fake_attempts=0 -- normal cold-start
    exploration, same as if this file didn't mention it at all. Only arms
    we actually found something for get the confidence boost.
    fake_attempts == 0 (the default) omits the column entirely, so this
    stays byte-for-byte the same soft-warm-start file it always was.
    """
    country_dir_name = country.replace(" ", "_")
    country_dir = os.path.join(out_dir, country_dir_name)
    os.makedirs(country_dir, exist_ok=True)
    out_path = os.path.join(country_dir, filename or f"{country_dir_name}.csv")

    rows = []
    for arm, q in sorted(averaged.items()):
        row = {"episode": 1, "time": 1, "action": arm, "q_value": round(q, 4)}
        if fake_attempts > 0:
            row["action_attempts"] = fake_attempts if q > 0 else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wrote {len(rows)} arm priors -> {out_path}"
          + (f" (fake_attempts={fake_attempts} on nonzero arms)" if fake_attempts > 0 else ""))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build per-country warm-start priors CSVs from past run(s)."
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Output directories of past runs to average, e.g. outputs27 outputs28")
    parser.add_argument("--countries", nargs="+", required=True,
                        help='Countries to build priors for, e.g. China "South Korea"')
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write <country>_priors.csv files into")
    parser.add_argument("--fake-attempts", type=int, default=0,
                        help="Hard warm-start: fake action_attempts count applied to arms with "
                             "q_value > 0 (0/default = soft warm-start, no action_attempts column)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for country in args.countries:
        print(f"\n=== {country} ===")
        averaged = average_across_runs(args.runs, country)
        if not averaged:
            print(f"  no data found for {country} in any given run; skipping")
            continue
        write_priors_csv(averaged, country, args.out_dir, fake_attempts=args.fake_attempts)


if __name__ == "__main__":
    main()
