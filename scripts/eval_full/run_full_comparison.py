#!/usr/bin/env python3
"""
Full 7-model comparison across China (GFWatch), Russia, and Kazakhstan.
10 episodes x 2000 steps per model per country.

Models:
  - UCB Naive            (baseline)
  - LinUCB               (contextual bandit)
  - HierTS cold          (hierarchical Thompson Sampling)
  - HierTS warm-start    (HierTS seeded from UCB Naive Q-values)
  - Uncertainty Sampling (active learning)
  - Query by Committee   (active learning)
  - QbC modAL            (active learning, incremental committee)

Usage (from project root):
    python3 scripts/run_full_comparison.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
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
from models.thompson_sampling.hier_ts import HierTS, HierTSActionSpace
from models.active_learning.uncertainty_sampling import UncertaintySampling
from models.active_learning.query_by_committee import QueryByCommittee
from models.active_learning.qbc_modal import QbCModAL

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
OUTPUT_DIR = os.path.join(ROOT, "models", "outputs", f"full_comparison_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Palette / styles ──────────────────────────────────────────────────────────

MODEL_STYLES = {
    "UCB Naive":            {"color": "#555555", "ls": "-",   "lw": 2.2},
    "LinUCB":               {"color": "#0072B2", "ls": "-",   "lw": 2.8},
    "HierTS (cold)":        {"color": "#CC79A7", "ls": ":",   "lw": 2.8},
    "HierTS (warm-start)":  {"color": "#D55E00", "ls": "--",  "lw": 3.2},
    "Uncertainty Sampling": {"color": "#009E73", "ls": "-.",  "lw": 2.2},
    "Query by Committee":   {"color": "#E69F00", "ls": "--",  "lw": 2.2},
    "QbC modAL":            {"color": "#56B4E9", "ls": ":",   "lw": 2.2},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def base_params(country_dir: str, name: str, gt_path: str,
                action_value_file=None) -> dict:
    return dict(
        target_feature             = "domain",
        num_episodes               = EPISODES,
        measurements_per_episode   = MEASUREMENTS,
        output_directory           = country_dir,
        outfile_csv                = os.path.join(country_dir, f"{name}.csv"),
        verbose                    = False,
        ground_truth_path          = gt_path,
        action_space_file          = ACTION_SPACE_FILE,
        action_value_file          = action_value_file,
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
country_results = {}   # country -> label -> df

for country, gt_path in COUNTRIES.items():
    print(f"\n{'='*65}")
    print(f"  {country}  ({EPISODES} eps × {MEASUREMENTS} steps)")
    print(f"{'='*65}")

    cdir = os.path.join(OUTPUT_DIR, country)
    os.makedirs(cdir, exist_ok=True)
    country_results[country] = {}

    # ── Phase 1: UCB Naive (needed for warm-start) ────────────────────────────
    df, cov, blk, auc = run_model(
        "UCB Naive", UCBNaive,
        {**base_params(cdir, "ucb_naive", gt_path),
         "c": 0.03, "step_size": 0.0, "initial_value_estimate": 0.0},
    )
    country_results[country]["UCB Naive"] = df
    all_rows.append({"Country": country, "Model": "UCB Naive",
                     "Final cov %": cov, "Blocked": blk, "AUC": auc})
    ucb_csv = os.path.join(cdir, "ucb_naive.csv")

    # ── Phase 2: remaining models ─────────────────────────────────────────────
    phase2 = [
        ("LinUCB", LinUCB,
         {**base_params(cdir, "linucb", gt_path),
          "alpha": 0.5, "initial_value_estimate": 0.0},
         {"action_space_klass": LinUCBActionSpace}),

        ("HierTS (cold)", HierTS,
         {**base_params(cdir, "hier_ts_cold", gt_path), "window": 0},
         {"action_space_klass": HierTSActionSpace}),

        ("HierTS (warm-start)", HierTS,
         {**base_params(cdir, "hier_ts_warm", gt_path,
                        action_value_file=ucb_csv), "window": 0},
         {"action_space_klass": HierTSActionSpace}),

        ("Uncertainty Sampling", UncertaintySampling,
         base_params(cdir, "uncertainty_sampling", gt_path),
         None),

        ("Query by Committee", QueryByCommittee,
         base_params(cdir, "query_by_committee", gt_path),
         None),

        ("QbC modAL", QbCModAL,
         base_params(cdir, "qbc_modal", gt_path),
         None),
    ]

    for label, klass, params, run_kwargs in phase2:
        df, cov, blk, auc = run_model(label, klass, params, run_kwargs)
        country_results[country][label] = df
        all_rows.append({"Country": country, "Model": label,
                         "Final cov %": cov, "Blocked": blk, "AUC": auc})

# ── Summary table ─────────────────────────────────────────────────────────────

summary = pd.DataFrame(all_rows)
summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
summary.to_csv(summary_path, index=False)

print(f"\n\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(summary.to_string(index=False))

# ── Plotting ──────────────────────────────────────────────────────────────────

sns.set_context("talk", font_scale=1.0)
sns.set_style("ticks")
plt.rcParams["pdf.fonttype"]  = 42
plt.rcParams["axes.linewidth"] = 1.4

ALL_LABELS = list(MODEL_STYLES.keys())


def plot_country(country: str, metric_key: str, ylabel: str,
                 filename: str, cumulative: bool = False, scale: float = 1.0):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label in ALL_LABELS:
        df  = country_results[country].get(label)
        if df is None:
            continue
        y = (get_cumulative_avg_across_episodes(df, metric_key)
             if cumulative else get_avg_across_episodes(df, metric_key))
        y = y * scale
        x = list(range(len(y)))
        auc = integrate(x[:MEASUREMENTS], y[:MEASUREMENTS].tolist())
        sty = MODEL_STYLES[label]
        ax.plot(x, y, label=f"{label}  (AUC={auc:,.0f})",
                color=sty["color"], linestyle=sty["ls"],
                linewidth=sty["lw"], alpha=0.92)

    ax.set_xlabel("Measurement step", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(country, fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor="#cccccc")
    sns.despine(offset=6)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{country}_{filename}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_combined_bar(metric: str, ylabel: str, filename: str):
    """Bar chart comparing all models across all countries on one metric."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, country in zip(axes, COUNTRIES):
        vals = [summary.loc[(summary.Country == country) &
                            (summary.Model == lbl), metric].values
                for lbl in ALL_LABELS]
        vals = [v[0] if len(v) else 0 for v in vals]
        colors = [MODEL_STYLES[l]["color"] for l in ALL_LABELS]
        bars = ax.bar(range(len(ALL_LABELS)), vals, color=colors, edgecolor="white",
                      linewidth=0.8)
        ax.set_title(country, fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(ALL_LABELS)))
        ax.set_xticklabels([l.replace(" (", "\n(") for l in ALL_LABELS],
                           fontsize=8, rotation=30, ha="right")
        ax.tick_params(labelsize=10)
        if ax == axes[0]:
            ax.set_ylabel(ylabel, fontsize=12)
    fig.suptitle(f"{ylabel} — 10 episodes × 2000 steps", fontsize=13)
    sns.despine()
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


for country in COUNTRIES:
    plot_country(country, "coverage",   "Coverage (%)",
                 "coverage.pdf", scale=100)
    plot_country(country, "is_blocked", "Cumulative blocked found",
                 "blocked.pdf",  cumulative=True)

plot_combined_bar("Final cov %", "Final Coverage (%)",    "combined_coverage.pdf")
plot_combined_bar("AUC",         "AUC (coverage)",        "combined_auc.pdf")
plot_combined_bar("Blocked",     "Avg blocked found",     "combined_blocked.pdf")

print(f"\nAll outputs: {OUTPUT_DIR}")
