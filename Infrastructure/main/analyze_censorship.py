"""
analyze_censorship.py

Parses CenRL output CSVs and produces presentation-ready charts showing:
  1. Overall blocking rate per country
  2. Top blocked categories globally
  3. Which categories are blocked in which countries (heatmap)
  4. Specific blocked domains by country

Usage:
    python analyze_censorship.py --output_dir /path/to/output20 --out plots/

The script scans for any *.csv files in output_dir (recursively), so it
works whether files are in subfolders or flat.
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from collections import defaultdict

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
PALETTE = sns.color_palette("Set2")

# ── helpers ────────────────────────────────────────────────────────────────

def load_all(output_dir: str) -> pd.DataFrame:
    """
    Load every *.csv in output_dir (recursively) and tag each row
    with its country name (derived from the folder / file name).
    """
    frames = []
    for path in glob.glob(os.path.join(output_dir, "**", "*.csv"),
                          recursive=True):
        fname = os.path.splitext(os.path.basename(path))[0]
        # The rolling "{country}.csv" already contains every row that also
        # appears in its "{country}_iter_NNN.csv" snapshots (frozen copies
        # taken at each soft-reset boundary) and in "{country}_measurements.csv"
        # (same data, different schema). Loading those too double/triple-counts
        # every measurement, so only the rolling file is the source of truth.
        if "_iter_" in fname or fname.endswith("_measurements"):
            continue
        # Strip trailing _iter_NNN so 'China_iter_001' -> 'China'
        country = fname.split("_iter_")[0].replace("_", " ")
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "is_blocked" not in df.columns:
            continue
        df["country"] = country
        # Clean up category label: "categories X" -> "X"
        if "action" in df.columns:
            df["category"] = (
                df["action"]
                .str.replace(r"^categories\s+", "", regex=True)
                .str.strip()
            )
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No valid CenRL CSVs found under '{output_dir}'"
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["is_blocked"] = combined["is_blocked"].astype(int)
    return combined


def blocking_rate(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("country")["is_blocked"]
        .mean()
        .sort_values(ascending=False)
    )


def top_blocked_categories(df: pd.DataFrame, n: int = 15) -> pd.Series:
    blocked = df[df["is_blocked"] == 1]
    return blocked["category"].value_counts().head(n)


def category_country_heatmap_data(df: pd.DataFrame,
                                   top_n: int = 20) -> pd.DataFrame:
    """
    Returns a DataFrame where rows = categories, cols = countries,
    values = fraction of measurements that were blocked for that
    category × country pair.
    """
    top_cats = (
        df[df["is_blocked"] == 1]["category"]
        .value_counts()
        .head(top_n)
        .index
    )
    subset = df[df["category"].isin(top_cats)]
    pivot = subset.pivot_table(
        index="category", columns="country",
        values="is_blocked", aggfunc="mean", fill_value=0
    )
    return pivot


def load_measurements(output_dir: str) -> pd.DataFrame:
    """
    Load every *_measurements.csv in output_dir (recursively), tagging each
    row with its country. This schema (target, arm, blocked, vp_count, ...)
    is flushed live per-measurement and carries vp_count and arm directly --
    neither exists in the rolling {country}.csv that load_all() uses, so this
    is a separate load path just for the confidence columns below.
    """
    frames = []
    for path in glob.glob(os.path.join(output_dir, "**", "*_measurements.csv"),
                          recursive=True):
        fname = os.path.splitext(os.path.basename(path))[0]
        country = fname[: -len("_measurements")].replace("_", " ")
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "blocked" not in df.columns:
            continue
        df["country"] = country
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def blocked_domains_with_confidence(measurements_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per blocked (country, target), with two confidence columns:
      - vp_confidence: vp_count for that measurement -- how many VPs actually
        voted on it (a single-VP result is much weaker evidence than a
        multi-VP majority agreeing).
      - rl_confidence: how many total measurements share this row's arm in
        this country -- how well-explored that category is for the model,
        i.e. is this block based on a well-tested arm or a rarely-tried one.
    """
    if measurements_df.empty:
        return pd.DataFrame(
            columns=["country", "target", "arm", "vp_confidence", "rl_confidence"]
        )

    arm_visits = (
        measurements_df.groupby(["country", "arm"])
        .size()
        .rename("rl_confidence")
        .reset_index()
    )
    blocked = measurements_df[measurements_df["blocked"] == 1].copy()
    blocked = blocked.merge(arm_visits, on=["country", "arm"], how="left")
    blocked = blocked.rename(columns={"vp_count": "vp_confidence"})
    # Same "categories X" -> "X" cleanup load_all() applies to the "action"
    # column -- here "arm" is the raw equivalent straight off the
    # measurements CSV (e.g. "categories Questionable Activities").
    blocked["category"] = (
        blocked["arm"].str.replace(r"^categories\s+", "", regex=True).str.strip()
    )
    return (
        blocked[["country", "target", "category", "vp_confidence", "rl_confidence"]]
        .drop_duplicates()
        .sort_values(["country", "target"])
    )


def add_measurement_num(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df ordered chronologically per country using (episode,
    time) — episode increments on each soft-reset, time resets to 1 within
    it, so sorting on both gives the true measurement order across resets —
    with a fresh measurement_num column (1..n per country). Shared by every
    over-time / exploration-order plot below so they don't each re-derive
    the same ordering.
    """
    ordered = df.sort_values(["country", "episode", "time"]).copy()
    ordered["measurement_num"] = ordered.groupby("country").cumcount() + 1
    return ordered


def censored_domains_found_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-country cumulative count of *unique* blocked domains discovered as
    measurements accumulate. Unlike a cumulative blocking *rate* (which can
    drift up or down as more measurements land), this is monotonically
    non-decreasing -- it answers "how many distinct censored sites has the
    model found so far for this many measurements spent," the same framing
    used to compare an active-learning approach's discovery efficiency
    against a baseline. Returns a long-form frame: country, measurement_num,
    domains_found.
    """
    ordered = add_measurement_num(df)

    def _cum_unique_blocked(group: pd.DataFrame) -> pd.Series:
        seen = set()
        counts = []
        for blocked, target in zip(group["is_blocked"], group["targets"]):
            if blocked and target not in seen:
                seen.add(target)
            counts.append(len(seen))
        return pd.Series(counts, index=group.index)

    ordered["domains_found"] = ordered.groupby(
        "country", group_keys=False
    ).apply(_cum_unique_blocked)
    return ordered[["country", "measurement_num", "domains_found"]]


def category_exploration_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-measurement (category, measurement_num, is_blocked, reward), pooled
    across every country. Categories/arms are shared action-space nodes, not
    country-specific, so pooling gives one combined picture of when the
    model tends to try each arm and how it paid off, rather than one chart
    per country.
    """
    ordered = add_measurement_num(df)
    return ordered[["category", "measurement_num", "is_blocked", "rewards"]]


def arm_health_checkpoints(df: pd.DataFrame, country: str,
                            checkpoints=(0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """
    For one country, how many times had each category/arm actually been
    tried by the 25/50/75/100% marks of that country's run -- entirely
    reconstructed from the rolling CSV's own chronological (episode, time)
    ordering, no live monitoring or run-time changes needed (the CSV is
    already a full replay log of every measurement the model made, in
    order, with which arm it hit each time).

    This is what would have flagged outputs36's "Digital Postcards" q_value
    swing ahead of time: an arm sitting at attempts=1 all the way through
    to the 100% checkpoint is one bad/noisy observation away from an
    extreme, misleading q_value with nothing to average it against -- the
    UCB exploration bonus (Sutton & Barto's np.inf-for-zero-attempts rule
    in ucb_naive.py) already tries to prevent this by prioritizing
    never-tried arms, so a low count that PERSISTS to the 75-100% marks
    means the model itself judged that branch low-priority given what it
    had learned by then, not that something malfunctioned -- worth seeing,
    not worth forcing.

    Returns a long-form frame: category, checkpoint_label (e.g. "25%"),
    cumulative_attempts, mean_reward -- pivot cumulative_attempts for a
    heatmap, or use both columns together for a flagged-arms report (low
    attempts alone isn't risky; low attempts AND a high mean_reward is).
    """
    country_df = df[df["country"] == country]
    ordered = add_measurement_num(country_df)
    total = ordered["measurement_num"].max()
    if pd.isna(total):
        return pd.DataFrame(columns=["category", "checkpoint_label", "cumulative_attempts"])

    # Cumulative attempt count per category, in chronological order --
    # same cumcount()+1 pattern add_measurement_num() itself uses.
    ordered = ordered.sort_values("measurement_num")
    ordered["cumulative_attempts"] = ordered.groupby("category").cumcount() + 1

    rows = []
    for frac in checkpoints:
        cutoff = int(round(frac * total))
        label = f"{int(round(frac * 100))}%"
        snapshot = ordered[ordered["measurement_num"] <= cutoff]
        # Last (i.e. highest) cumulative_attempts seen per category by this
        # checkpoint == that category's total attempts so far. Mean reward
        # over that same snapshot is what actually makes a low sample count
        # risky or not -- a category sitting at attempts=1 with reward=0.0
        # isn't claiming anything alarming; attempts=1 with reward=1.0 is
        # exactly the "one flipped bit, nothing to average it against" case.
        per_category = snapshot.groupby("category").agg(
            cumulative_attempts=("cumulative_attempts", "max"),
            mean_reward=("rewards", "mean"),
        )
        for category, row in per_category.iterrows():
            rows.append({"category": category, "checkpoint_label": label,
                        "cumulative_attempts": int(row["cumulative_attempts"]),
                        "mean_reward": row["mean_reward"]})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Categories never reached by a given checkpoint just don't appear in
    # that checkpoint's snapshot -- fill those combinations in as 0 so the
    # heatmap/report don't silently drop them.
    all_categories = ordered["category"].unique()
    all_labels = [f"{int(round(f * 100))}%" for f in checkpoints]
    full_index = pd.MultiIndex.from_product([all_categories, all_labels],
                                             names=["category", "checkpoint_label"])
    result = (
        result.set_index(["category", "checkpoint_label"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return result


# ── plots ──────────────────────────────────────────────────────────────────

def plot_blocking_rate(rates: pd.Series, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[2] if r > 0.05 else PALETTE[0]
              for r in rates.values]
    bars = ax.barh(rates.index[::-1], rates.values[::-1] * 100,
                   color=colors[::-1], edgecolor="white", height=0.6)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("% of Measurements Blocked", fontsize=12)
    ax.set_title("Blocking Rate by Country", fontsize=14, fontweight="bold",
                 pad=12)

    # annotate bars
    for bar, val in zip(bars, rates.values[::-1]):
        if val > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=9)

    ax.axvline(rates.mean() * 100, color="grey", linestyle="--",
               linewidth=1, label=f"Mean: {rates.mean()*100:.1f}%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_top_categories(counts: pd.Series, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("flare", len(counts))[::-1]
    ax.barh(counts.index[::-1], counts.values[::-1],
            color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Number of Blocked Measurements", fontsize=12)
    ax.set_title("Most Frequently Blocked Categories (All Countries)",
                 fontsize=14, fontweight="bold", pad=12)
    for i, (idx, val) in enumerate(zip(counts.index[::-1],
                                        counts.values[::-1])):
        ax.text(val + 0.1, i, str(val), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_heatmap(pivot: pd.DataFrame, out_path: str):
    if pivot.empty:
        return
    fig_h = max(6, len(pivot) * 0.4)
    fig_w = max(8, len(pivot.columns) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd", linewidths=0.4, linecolor="white",
        annot=True, fmt=".0%",
        cbar_kws={"label": "Block Rate", "shrink": 0.6},
        vmin=0, vmax=1,
    )
    ax.set_title("Blocking Rate by Category × Country",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_censored_domains_found(df: pd.DataFrame, out_path: str):
    """
    Cumulative unique censored domains found vs. measurements spent, one
    line per country, each labeled with its AUC (trapezoidal area under its
    own curve) -- a higher AUC means that country's run found its censored
    domains more efficiently per measurement.
    """
    trend = censored_domains_found_over_time(df)
    if trend.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    countries = sorted(trend["country"].unique())
    colors = sns.color_palette("husl", len(countries))
    for color, country in zip(colors, countries):
        sub = trend[trend["country"] == country]
        trapz = getattr(np, "trapezoid", None) or np.trapz
        auc = trapz(sub["domains_found"], sub["measurement_num"])
        ax.plot(sub["measurement_num"], sub["domains_found"],
                label=f"{country} (auc = {auc:,.0f})",
                color=color, linewidth=1.6)
    ax.set_xlabel("Number of Measurements", fontsize=12)
    ax.set_ylabel("Censored Websites Found", fontsize=12)
    ax.set_title("Cumulative Censored Domains Found Over Time", fontsize=14,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1))
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_category_exploration(df: pd.DataFrame, out_path: str):
    """
    Per-category (arm) exploration diagnostic, combining three signals:
      - box plot: distribution of measurement_num values this arm was
        pulled at (categories ordered left-to-right by mean measurement_num,
        so arms the model tends to explore early appear first)
      - scatter: individual pulls colored by whether that measurement came
        back blocked
      - red diamonds (secondary axis): average reward per category
    """
    data = category_exploration_data(df)
    if data.empty:
        return

    order = (
        data.groupby("category")["measurement_num"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    plot_data = data.assign(
        Blocked=data["is_blocked"].map({0: "False", 1: "True"})
    )

    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.5), 7))
    sns.boxplot(
        data=plot_data, x="category", y="measurement_num", order=order,
        ax=ax, fliersize=0, linewidth=1,
        # Separate edge color from the light grey fill: a category with a
        # very narrow IQR (e.g. most measurements clustered in a tight
        # window) collapses to a near-zero-height box, and a lightgrey
        # edge on a sub-pixel-tall box becomes invisible against white.
        # A dark edge stays visible as a line even when the box has no
        # perceptible height.
        boxprops=dict(facecolor="lightgrey", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )
    sns.stripplot(
        data=plot_data, x="category", y="measurement_num", order=order,
        hue="Blocked",
        # Explicit high-contrast colors rather than PALETTE[0]/[2] -- those
        # Set2 tones read too close to the grey boxes to stand out. Purple
        # for True is the whole point of this chart (spotting blocked
        # outcomes at a glance), so it needs to pop against both the grey
        # boxes and the blue "False" dots.
        palette={"False": "#4C72B0", "True": "#9932CC"},
        ax=ax, size=4, alpha=0.85, jitter=0.25, linewidth=0.3, edgecolor="white",
    )
    ax.set_xlabel("Categories (Arms)", fontsize=12)
    ax.set_ylabel("Measurement Step", fontsize=12)
    ax.legend(title="Blocked?", fontsize=8, loc="upper left")

    ax2 = ax.twinx()
    reward_means = plot_data.groupby("category")["rewards"].mean().reindex(order)
    ax2.plot(range(len(order)), reward_means.values, "D", color="red",
             markersize=5, label="Average Reward")
    ax2.set_ylabel("Average Reward", color="red", fontsize=12)
    ax2.tick_params(axis="y", colors="red")

    ax.set_title("Category Exploration: When Each Arm Was Tried, and Outcome",
                 fontsize=14, fontweight="bold", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_arm_health_checkpoints(health_df: pd.DataFrame, country: str, out_path: str,
                                 low_sample_threshold: int = 5):
    """
    Heatmap: category x checkpoint (25/50/75/100%), cell = cumulative
    attempts for that arm by that point in the run -- numbers, not just
    color, same reasoning as plot_category_qvalues.py's heatmap: color
    alone can't be read precisely, and the whole point here is the actual
    count. Categories sorted by final (100%) attempt count descending, so
    well-explored arms cluster at the top and under-explored ones (the
    ones actually worth worrying about) sink to the bottom where they're
    easy to scan as a block.

    Also prints a flagged-arms list to the console: a category still under
    low_sample_threshold attempts by the second-to-last checkpoint (75% by
    default) AND showing a nonzero mean reward there -- not low attempts
    alone, since most categories in a ~95-way action space naturally only
    get tried once or twice and that's normal, not risky. A low-sample arm
    sitting at reward 0.0 isn't claiming anything alarming; low attempts
    with a high reward is exactly the "one noisy observation, nothing to
    average it against" situation that let outputs36's single bad "Digital
    Postcards" pull swing its q_value to the extreme.
    """
    if health_df.empty:
        return

    checkpoint_order = list(dict.fromkeys(health_df["checkpoint_label"]))  # first-seen order
    pivot = health_df.pivot(index="category", columns="checkpoint_label", values="cumulative_attempts")
    pivot = pivot[checkpoint_order]
    reward_pivot = health_df.pivot(index="category", columns="checkpoint_label", values="mean_reward")
    final_col = checkpoint_order[-1]
    pivot = pivot.sort_values(final_col, ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, len(checkpoint_order) * 1.5), max(6, len(pivot) * 0.22)))
    sns.heatmap(
        pivot, ax=ax, cmap="Purples", linewidths=0.4, linecolor="white",
        annot=True, fmt="d", annot_kws={"fontsize": 7},
        cbar_kws={"label": "Cumulative Attempts"},
    )
    ax.set_xlabel("Run Progress", fontsize=12)
    ax.set_ylabel("Category (Arm)", fontsize=12)
    ax.set_title(f"{country}: Arm Health Over Time (Attempts per Checkpoint)",
                 fontsize=14, fontweight="bold", pad=12)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    if len(checkpoint_order) >= 2:
        review_col = checkpoint_order[-2]  # e.g. "75%" when checkpoints are 25/50/75/100
        low_sample = pivot[review_col] < low_sample_threshold
        nonzero_reward = reward_pivot[review_col] > 0
        flagged = pivot[low_sample & nonzero_reward].index.tolist()
        if flagged:
            flagged.sort(key=lambda c: reward_pivot.loc[c, review_col], reverse=True)
            print(f"\n  {country}: {len(flagged)} arm(s) under {low_sample_threshold} attempts "
                  f"AND showing nonzero reward by the {review_col} mark -- worth a second look, "
                  f"one noisy observation is enough to be driving these:")
            for cat in flagged:
                print(f"    {cat}: {int(pivot.loc[cat, review_col])} attempts / "
                      f"reward {reward_pivot.loc[cat, review_col]:.2f} at {review_col}  ->  "
                      f"{int(pivot.loc[cat, final_col])} attempts / "
                      f"reward {reward_pivot.loc[cat, final_col]:.2f} at {final_col}")
        else:
            print(f"\n  {country}: no arms flagged -- every category with meaningful reward "
                  f"by the {review_col} mark has at least {low_sample_threshold} attempts backing it up.")


def plot_blocked_domains_table(confidence_df: pd.DataFrame, out_path: str):
    """
    Text-based table: country | blocked domain | category | VP votes | RL arm visits.

    VP votes = how many vantage points actually voted on this result (a
    3-VP unanimous result is solid evidence; a 1-VP result has no cross-check
    at all). RL arm visits = how many total measurements share this domain's
    arm/category, i.e. how well-explored that arm is for the model. Category
    makes it obvious at a glance what kind of content each block came from.
    """
    if confidence_df.empty:
        return

    rows = []
    for _, r in confidence_df.iterrows():
        rows.append(
            f"  {r['country']:<16} {r['target']:<32} {r['category']:<28} "
            f"{int(r['vp_confidence']):<10} {int(r['rl_confidence'])}"
        )

    fig, ax = plt.subplots(figsize=(16, max(4, len(rows) * 0.28)))
    ax.axis("off")
    text = "Blocked Domains by Country\n\n"
    text += (f"  {'Country':<16} {'Domain':<32} {'Category':<28} "
              f"{'VP Votes':<10} {'RL Arm Visits'}\n")
    text += "  " + "-" * 98 + "\n"
    text += "\n".join(rows)
    ax.text(0.01, 0.99, text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", family="monospace")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def extract_run_label(output_dir: str) -> str:
    """
    Best-effort run number from the output directory name, e.g.
    '.../outputs29' -> '29'. Falls back to the full basename if no trailing
    digits are found (so plots from a non-numbered directory still get a
    distinguishing prefix rather than silently having none).
    """
    basename = os.path.basename(os.path.normpath(output_dir))
    match = re.search(r"(\d+)$", basename)
    return match.group(1) if match else basename


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze CenRL censorship results")
    parser.add_argument("--output_dir", default="/home/claude/cenrl_outputs/output20",
                        help="Directory containing country CSV files")
    parser.add_argument("--out", default="/home/claude/plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    run_label = extract_run_label(args.output_dir)
    def plot_path(name: str) -> str:
        return os.path.join(args.out, f"{run_label}-{name}")

    print(f"Loading CSVs from: {args.output_dir}")
    df = load_all(args.output_dir)
    print(f"Loaded {len(df):,} rows across "
          f"{df['country'].nunique()} countries: "
          f"{sorted(df['country'].unique())}")

    # ── summary stats ──────────────────────────────────────────────────
    rates = blocking_rate(df)
    print("\n── Blocking rates ──")
    for country, rate in rates.items():
        bar = "█" * int(rate * 40)
        print(f"  {country:<20} {rate*100:5.1f}%  {bar}")

    total_blocked = df["is_blocked"].sum()
    total = len(df)
    print(f"\nTotal: {total_blocked} blocked / {total} measured "
          f"({total_blocked/total*100:.1f}% overall)")

    # ── plots ──────────────────────────────────────────────────────────
    plot_blocking_rate(
        rates,
        plot_path("01_blocking_rate_by_country.png")
    )

    cats = top_blocked_categories(df, n=15)
    if not cats.empty:
        plot_top_categories(
            cats,
            plot_path("02_top_blocked_categories.png")
        )

    pivot = category_country_heatmap_data(df, top_n=20)
    if not pivot.empty:
        plot_heatmap(
            pivot,
            plot_path("03_category_country_heatmap.png")
        )

    plot_censored_domains_found(
        df,
        plot_path("04_censored_domains_found_over_time.png")
    )

    measurements_df = load_measurements(args.output_dir)
    confidence_df = blocked_domains_with_confidence(measurements_df)
    if not confidence_df.empty:
        plot_blocked_domains_table(
            confidence_df,
            plot_path("05_blocked_domains_table.png")
        )

    plot_category_exploration(
        df,
        plot_path("06_category_exploration.png")
    )

    # Arm health over time, for whichever country actually has real signal
    # (highest blocking rate -- same country every other "what's actually
    # going on here" chart in this script already focuses on).
    if not rates.empty and rates.iloc[0] > 0:
        top_country = rates.index[0]
        health_df = arm_health_checkpoints(df, top_country)
        if not health_df.empty:
            plot_arm_health_checkpoints(
                health_df, top_country,
                plot_path("07_arm_health_checkpoints.png")
            )

    print(f"\nAll plots saved to: {args.out}/")


if __name__ == "__main__":
    main()

'''
The command that we use to run the script:
conda activate cenv
python3 analyze_censorship.py \
  --output_dir /home/cenrl/cenrl_outputs/initial_tests/outputs20 \
  --out /home/cenrl/cenrl_outputs/plots/

Actually, new and improved command, directs plots to output file itself:
cd /home/cenrl/cenrl/Infrastructure/main
python3 analyze_censorship.py \
  --output_dir /home/cenrl/cenrl_outputs/initial_tests/outputs23 \
  --out /home/cenrl/cenrl_outputs/initial_tests/outputs23


'''