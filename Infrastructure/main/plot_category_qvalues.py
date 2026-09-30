"""
plot_category_qvalues.py

Standalone add-on script -- by default, ONE heatmap: countries as rows,
categories (arms) as columns, cell color/value = that country's final
learned q_value for that category. Point it at any run's output directory
(e.g. outputs29, or the priors/ folder) and it scans every country
subfolder it finds, so you can see every country's q_values at once instead
of flipping between per-country charts.

Reuses the exact same final-value extraction as build_warm_start_priors.py's
extract_final_arm_values() (which itself mirrors read_action_value_file() in
models/base/action_space.py:146-159): sort by (episode, time), take the last
q_value per (episode, action), then mean across episodes.

Usage:
    python3 plot_category_qvalues.py --run /path/to/outputs29 --out /path/to/out_dir
    python3 plot_category_qvalues.py --run /home/cenrl/cenrl_outputs/priors --out /tmp/priors_charts
    # per-country bar charts instead of the combined heatmap:
    python3 plot_category_qvalues.py --run /path/to/outputs29 --out /path/to/out_dir --per-country
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_final_arm_values(csv_path: str) -> dict:
    """One country's final learned q_value per arm/category."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["episode", "time"])
    last_occurrences = df.groupby(["episode", "action"]).last().reset_index()
    action_averages = (
        last_occurrences.groupby("action")[["q_value"]].mean().reset_index()
    )
    return action_averages.set_index("action")["q_value"].to_dict()


def find_country_csvs(run_dir: str) -> dict:
    """country_name -> csv path, for every <Country>/<Country>.csv under run_dir."""
    found = {}
    for entry in sorted(Path(run_dir).iterdir()):
        if not entry.is_dir():
            continue
        csv_path = entry / f"{entry.name}.csv"
        if csv_path.exists():
            found[entry.name.replace("_", " ")] = str(csv_path)
    return found


def plot_country_qvalues(country: str, values: dict, out_path: str):
    clean = {
        re.sub(r"^categories\s+", "", arm).strip(): q
        for arm, q in values.items()
    }
    ordered = dict(sorted(clean.items(), key=lambda kv: kv[1], reverse=True))
    categories = list(ordered.keys())
    qvals = list(ordered.values())

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.35), 6))
    colors = ["#9932CC" if q > 0 else "#B0B0B0" for q in qvals]
    ax.bar(categories, qvals, color=colors)
    ax.set_xlabel("Category (Arm)", fontsize=12)
    ax.set_ylabel("Final Q-Value", fontsize=12)
    ax.set_title(f"{country}: Final Q-Value by Category", fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def build_qvalue_matrix(country_csvs: dict) -> pd.DataFrame:
    """country x category matrix of final q_values. Missing (country, category)
    pairs -- a category that country never explored -- fill as 0.0, matching
    get_default_q_value()'s cold-start default rather than NaN, since an
    untried arm and a tried-but-zero-reward arm are visually indistinguishable
    here."""
    rows = {}
    for country, csv_path in country_csvs.items():
        values = extract_final_arm_values(csv_path)
        clean = {
            re.sub(r"^categories\s+", "", arm).strip(): q
            for arm, q in values.items()
        }
        rows[country] = clean

    matrix = pd.DataFrame(rows).T.fillna(0.0)
    # Sort columns by the strongest signal seen for that category across any
    # country, descending -- so real signal clusters on the left instead of
    # being scattered among ~95 all-zero columns.
    matrix = matrix[matrix.max(axis=0).sort_values(ascending=False).index]
    return matrix


def plot_qvalue_heatmap(matrix: pd.DataFrame, out_path: str, show_all_zero_cols: bool = False):
    """Numbers-first heatmap: color still encodes magnitude, but every
    nonzero cell is also annotated with its actual q_value so you don't have
    to eyeball shading. All-zero columns (a category no country ever found
    signal in) are dropped by default -- with ~95 categories and mostly 0.0
    everywhere, keeping them just pushes cells too narrow to hold text."""
    if not show_all_zero_cols:
        matrix = matrix.loc[:, matrix.max(axis=0) > 0]

    annot = matrix.applymap(lambda v: f"{v:.2f}" if v > 0 else "")

    fig, ax = plt.subplots(figsize=(max(12, matrix.shape[1] * 0.7), max(4, matrix.shape[0] * 0.7)))
    sns.heatmap(
        matrix, ax=ax, cmap="Purples", linewidths=0.4, linecolor="white",
        annot=annot, fmt="", annot_kws={"fontsize": 7},
        cbar_kws={"label": "Final Q-Value"},
    )
    ax.set_xlabel("Category (Arm)", fontsize=12)
    ax.set_ylabel("Country", fontsize=12)
    ax.set_title("Final Q-Value by Country x Category", fontsize=14, fontweight="bold", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Chart final q_value by category across every country in a run directory."
    )
    parser.add_argument("--run", required=True, help="Run output directory, e.g. outputs29 or the priors/ folder")
    parser.add_argument("--out", required=True, help="Directory to write chart(s) into")
    parser.add_argument("--per-country", action="store_true",
                        help="Make one bar chart per country instead of the combined heatmap")
    parser.add_argument("--show-all-zero-cols", action="store_true",
                        help="Keep categories with 0.0 for every country instead of dropping them")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    country_csvs = find_country_csvs(args.run)
    if not country_csvs:
        print(f"No <Country>/<Country>.csv files found under {args.run}")
        return

    if args.per_country:
        for country, csv_path in country_csvs.items():
            values = extract_final_arm_values(csv_path)
            if not values:
                print(f"{country}: no arm values found, skipping")
                continue
            out_name = country.replace(" ", "_") + "_qvalues.png"
            plot_country_qvalues(country, values, os.path.join(args.out, out_name))
        return

    matrix = build_qvalue_matrix(country_csvs)
    plot_qvalue_heatmap(
        matrix, os.path.join(args.out, "all_countries_qvalues_heatmap.png"),
        show_all_zero_cols=args.show_all_zero_cols,
    )


if __name__ == "__main__":
    main()
