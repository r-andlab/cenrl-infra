#!/usr/bin/env python3
"""
Rerun UCB Naive and LinUCB with reverted settings:
  - FEATURES = ["categories"]
  - UCB Naive: c=0.03, step_size=0.0, initial_value=0.0
  - LinUCB:    alpha=0.5, initial_value=0.0

10 episodes x 2000 steps across China, Russia, Kazakhstan.

Usage (from project root):
    python3 scripts/run_ucb_linucb_rerun.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from baselines.utils import integrate
from common.utils import get_avg_across_episodes, get_cumulative_avg_across_episodes
from models.base.model import create_and_run_model
from models.ucb.ucb_naive import UCBNaive
from models.contextual_bandit.linucb import LinUCB, LinUCBActionSpace

# ── Config ────────────────────────────────────────────────────────────────────

EPISODES     = 10
MEASUREMENTS = 2000
FEATURES     = ["categories"]

ACTION_SPACE_FILE = os.path.join(ROOT, "inputs", "tranco",
                                 "tranco_categories_subdomain_tld_entities_top10k.csv")

COUNTRIES = {
    "CHINA":      os.path.join(ROOT, "inputs", "gfwatch",    "gfwatch-blocklist.csv"),
    "RUSSIA":     os.path.join(ROOT, "inputs", "russia",     "russia-blocklist.csv"),
    "KAZAKHSTAN": os.path.join(ROOT, "inputs", "kazakhstan", "kazakhstan-blocklist.csv"),
}

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(ROOT, "models", "outputs", f"ucb_linucb_rerun_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_STYLES = {
    "UCB Naive": {"color": "#555555", "ls": "-",  "lw": 2.2},
    "LinUCB":    {"color": "#0072B2", "ls": "-.", "lw": 2.8},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def base_params(country_dir, name, gt_path):
    return dict(
        target_feature             = "domain",
        num_episodes               = EPISODES,
        measurements_per_episode   = MEASUREMENTS,
        output_directory           = country_dir,
        outfile_csv                = os.path.join(country_dir, f"{name}.csv"),
        verbose                    = False,
        ground_truth_path          = gt_path,
        action_space_file          = ACTION_SPACE_FILE,
        action_value_file          = None,
        features                   = FEATURES,
        consider_unknown           = "Empty",
        sample_by_target_rank      = False,
        action_space_multi_parents = False,
        num_of_processes_for_episodes = 1,
    )


def run_model(label, klass, params, run_kwargs=None):
    print(f"  Running: {label}")
    df = create_and_run_model(
        klass, params,
        addition_model_run_kwargs=run_kwargs or None,
    )
    cov = df.groupby("episode")["coverage"].last().mean()
    blk = df.groupby("episode")["is_blocked"].sum().mean()
    cov_series = get_avg_across_episodes(df, "coverage")
    x = list(range(len(cov_series)))
    auc = integrate(x[:MEASUREMENTS], (cov_series * 100).tolist()[:MEASUREMENTS])
    print(f"    → cov={cov*100:.1f}%  blocked={blk:.1f}  AUC={auc:,.0f}")
    return df, round(cov * 100, 1), round(blk, 1), round(auc, 0)


# ── Main loop ─────────────────────────────────────────────────────────────────

all_rows = []
country_results = {}

for country, gt_path in COUNTRIES.items():
    print(f"\n{'='*65}")
    print(f"  {country}  ({EPISODES} eps × {MEASUREMENTS} steps)")
    print(f"{'='*65}")

    cdir = os.path.join(OUTPUT_DIR, country)
    os.makedirs(cdir, exist_ok=True)
    country_results[country] = {}

    df, cov, blk, auc = run_model(
        "UCB Naive", UCBNaive,
        {**base_params(cdir, "ucb_naive", gt_path),
         "c": 0.03, "step_size": 0.0, "initial_value_estimate": 0.0},
    )
    country_results[country]["UCB Naive"] = df
    all_rows.append({"Country": country, "Model": "UCB Naive",
                     "Final cov %": cov, "Blocked": blk, "AUC": auc})

    df, cov, blk, auc = run_model(
        "LinUCB", LinUCB,
        {**base_params(cdir, "linucb", gt_path),
         "alpha": 0.5, "initial_value_estimate": 0.0},
        run_kwargs={"action_space_klass": LinUCBActionSpace},
    )
    country_results[country]["LinUCB"] = df
    all_rows.append({"Country": country, "Model": "LinUCB",
                     "Final cov %": cov, "Blocked": blk, "AUC": auc})

# ── Plots ─────────────────────────────────────────────────────────────────────

sns.set_context("talk", font_scale=1.1)
sns.set_style("ticks")
plt.rcParams["pdf.fonttype"]  = 42
plt.rcParams["axes.linewidth"] = 1.4

ALL_LABELS = ["UCB Naive", "LinUCB"]

for country in COUNTRIES:
    for metric, ylabel, cumulative, scale, fname in [
        ("is_blocked", "Cumulative Blocked Found", True,  1.0,   "blocked.pdf"),
        ("coverage",   "Coverage (%)",             False, 100.0, "coverage.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for label in ALL_LABELS:
            df  = country_results[country][label]
            y   = (get_cumulative_avg_across_episodes(df, metric)
                   if cumulative else get_avg_across_episodes(df, metric))
            y   = y * scale
            x   = list(range(len(y)))
            auc = integrate(x[:MEASUREMENTS], y[:MEASUREMENTS].tolist())
            style = MODEL_STYLES[label]
            ax.plot(x, y, label=f"{label} (AUC={auc:,.0f})",
                    color=style["color"], linestyle=style["ls"],
                    linewidth=style["lw"], alpha=0.92)

        ax.set_title(country, fontsize=14)
        ax.set_xlabel("Measurement step", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10, loc="lower right", framealpha=0.9, edgecolor="#cccccc")
        sns.despine(offset=6)
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"{country.lower()}_{fname}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

# ── Summary ───────────────────────────────────────────────────────────────────

summary = pd.DataFrame(all_rows)
summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

print(f"\n\n{'='*65}")
print("SUMMARY")
print('='*65)
print(summary.to_string(index=False))
print(f"\nAll outputs: {OUTPUT_DIR}")
