# main/

Contains the primary runtime logic: the orchestrator, per-country regional nodes, and the vantage point pool manager.

---

## Modules

### `orchestrator.py` — `Orchestrator`

The central coordinator that manages the full measurement campaign lifecycle.

**Constructor parameters:**

| Parameter | Type | Description |
|---|---|---|
| `params` | `Dict` | Model/parser parameters (passed through to nodes) |
| `vantage_points` | `VantagePoints` | Shared VP pool manager (replaces the old hardcoded map) |
| `go_api_endpoint` | `str` | URL of the Go Hyperquack service |
| `services` | `List[str]` | Services to test (e.g. `["https"]`) |
| `countries` | `List[str]` | Target countries (defaults to all in CSV) |
| `vps_per_country` | `int` | How many VPs to draw per country (default 3) |
| `debug` | `bool` | Run without a real Go server |

**Tick loop (`tick()`):**

1. **Process VP evaluations** — Drains `EvalStore`, confirms or rejects VPs, creates `RegionalNode`s lazily on first good VP per country.
2. **Drain raw results** — Fetches `TestPayload`s from the measurement store.
3. **Check VP health** — Tracks `controls_failed` counts; removes VPs exceeding `VP_FAILURE_THRESHOLD` (default 5).
4. **Feed aggregator** — Sends per-VP results into `MeasurementAggregator`.
5. **Deliver aggregated results** — Passes majority-vote finalized results to each node's model.
6. **Schedule new measurements** — Gets target URLs from the node, schedules across all active VPs.

**VP lifecycle:**
- On startup, `_bootstrap_vps()` draws VPs and sends them for Go server evaluation.
- Good VPs are confirmed active; bad VPs are rejected and automatically replaced.
- During measurements, unhealthy VPs are detected and replaced.

### `node.py` — `RegionalNode`

A per-country agent that wraps a `BatchUCB` reinforcement learning model.

**Key responsibilities:**
- Manages episode lifecycle (multiple episodes per country)
- Tracks in-flight measurement count (number of **targets**, not per-VP measurements)
- Provides `maybe_request_more()` to get the next batch of target URLs
- Provides `maybe_update_model(measurements)` to absorb aggregated results

**Note:** `RegionalNode` no longer manages VP state directly. VP active/inactive tracking is handled entirely by the `VantagePoints` class.

### `vantage_points.py` — `VantagePoints`

Thread-safe pool manager for per-country vantage point IP addresses. This is the **single source of truth** for VP state, shared between the orchestrator, server, and funneler.

**Initialization:**
- Parses an `ev_file` CSV (columns: `ipv4`, `ipv6`, `country`, ...)
- Applies a CIDR `blocklist_file` — entries like `10.0.0.0/8` filter any IP in that range
- Trims each country's pool to `max_size`

**Per-country state:**
- `active` set — VPs currently in use for measurements
- `inactive` set — candidates available for future use

**Key methods:**

| Method | Purpose |
|---|---|
| `get_vantage(country)` | Draw one random inactive VP, move to active |
| `get_n_vantages(country, n)` | Draw up to n inactive VPs, move to active |
| `confirm_active(country, vp)` | VP passed evaluation, ensure it stays active |
| `reject_vp(country, vp)` | VP failed, remove entirely, draw and return replacement |
| `evaluate(country, vp, ok)` | Legacy method: ok=True moves to inactive, ok=False removes |
| `get_active(country)` | List currently active VPs |
| `get_inactive(country)` | List currently inactive VPs |
| `countries()` | List all countries with VP candidates |

### `test_vantage_points.py`

Unit tests for `VantagePoints`. Run with:
```
python -m unittest Infrastructure.main.test_vantage_points -v
```

---
