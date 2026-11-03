# Infrastructure plotting scripts

Reusable matplotlib + seaborn CLIs for visualizing Phase 3 output streams.

## Inputs

These scripts assume the layout produced by `Infrastructure/main/orchestrator.py`:

    <run-dir>/
    ├── run_config.json            (optional — used for plot titles)
    ├── tick_timings.csv           (orchestrator per-tick instrumentation)
    └── <country>/
        └── <country>_measurements.csv   (per-measurement, flushed live)

All scripts default their output to `<run-dir>/plots/` (run-level) or `<run-dir>/<country>/plots/` (per-country). Override with `--out`.

## Scripts

### plot_tick_timings.py

Visualizes `tick_timings.csv` for Python-vs-Go bottleneck analysis.

Produces:

- `tick_timings_stacked.png` — stacked area of `drain / eval_processing / aggregate / model_update / schedule` over `tick_idx`.
- `tick_timings_total.png` — total tick time with rolling-mean overlay.
- `tick_timings_distributions.png` — per-substep histograms (2x3 grid, with KDE).
- `tick_timings_vs_active_countries.png` — total tick time scattered against active country count (regression overlay if >1 unique x).

Example:

    python3 Infrastructure/scripts/plotting/plot_tick_timings.py --run-dir outputs/run1

### plot_measurements.py

Visualizes per-country `<country>_measurements.csv` for learning/quality analysis.

Per country, produces: `reward_cumulative_monotonic.png`, `reward_cumulative_episode_mean.png`, `reward_rolling_monotonic.png`, `reward_rolling_episode_mean.png`, `blocked_rate_rolling_monotonic.png`, `blocked_rate_rolling_episode_mean.png`, `latency_distribution.png`, `vp_count_distribution.png`, `coverage_over_time_monotonic.png`, `coverage_over_time_episode_mean.png`, `optimal_arm_rate_monotonic.png`, `optimal_arm_rate_episode_mean.png`, `q_value_top_arms_monotonic.png`, `q_value_top_arms_episode_mean.png`, `arm_counts.png`.

Each per-country figure is wrapped in try/except so one broken plot does not skip the rest.

Example:

    python3 Infrastructure/scripts/plotting/plot_measurements.py --run-dir outputs/run1 --top-arms 10 --rolling 100

### plot_accuracy.py

Joins each country's `<country>_measurements.csv` against a ground-truth blocklist and emits accuracy plots.

Per country, produces: `accuracy_summary.png` (precision/recall/F1/accuracy bars), `confusion_matrix.png` (2x2 tn/fp/fn/tp heatmap), `f1_over_time_monotonic.png` (rolling precision/recall/F1 vs `measurement_idx`), `f1_over_time_episode_mean.png` (within-episode rolling P/R/F1, mean +/- std across episodes by `step_idx`).

Exits non-zero with a clear stderr message if the ground-truth file parses to an empty set.

Example:

    python3 Infrastructure/scripts/plotting/plot_accuracy.py --run-dir outputs/run1 --ground-truth inputs/gfwatch/gfwatch-blocklist.csv

## Ground-truth file format

Two formats are accepted (sniffed from the first non-empty line):

1. **CSV with `domain` header** (matches the format under `inputs/gfwatch/`, `inputs/russia/`, `inputs/kazakhstan/`):

       domain
       example.com
       blocked-site.net
       *.wildcard.example

   A leading `*.` is stripped on load (Russia-style wildcard rows match the bare domain).

2. **Plain text, one domain per line.** Blank lines and lines starting with `#` are ignored.

All comparisons are case-insensitive (both the ground-truth set and the `target` column are lowercased before joining).

## Style

All scripts use the repo's existing paper-plot style (see `scripts/plot_utils.py`): seaborn paper context (`font_scale=1.5`), `ticks` style, `pdf.fonttype = 42` for matplotlib PDF embedding. Outputs are PNG at `dpi=150`.

## Notes

- Per-step charts emit two variants: `*_monotonic.png` (x-axis = `measurement_idx` 0..N-1, rows sorted by `utc_timestamp` ascending — shows the full run as one continuous trajectory) and `*_episode_mean.png` (x-axis = `step_idx`, mean +/- std band aggregated across episodes — shows the average per-episode learning curve). Single-episode runs still produce both; the episode-mean band degenerates to a flat line with no shading. Histograms, bar charts, and run-end aggregates keep their single-PNG names because they have no temporal x-axis.
- Scripts work on partial runs — they only require the columns Phase 3 produces and skip per-country dirs that are empty or missing the measurements CSV.
- Per-figure failures are caught and reported; one broken figure does not abort the rest of the country's plots.
- `run_config.json` is optional — if absent, plot titles drop the git-SHA suffix.
- These scripts live under `Infrastructure/scripts/plotting/` and are distinct from the legacy `scripts/` plotters at repo root (those are for the controlled-evaluation models in `models/`).
- Scope: single-run analysis only. Multi-run comparison is out of scope.
