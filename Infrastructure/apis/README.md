# apis/

Defines the API and server layer for communication between the Python orchestration system and the Go-based Hyperquack measurement service.

---

## Modules

### `api.py`

Base abstract class `Api` with a utility method `fix_target()` that normalizes and validates URLs (handles missing schemes, lowercasing, default port removal, path normalization). Subclassed by `HyperQuackAPI`.

### `server.py` — `MeasurementReceiver`

A FastAPI server (default port 9000) that receives callbacks from the Go server. Runs in a background daemon thread.

**Endpoints:**

| Endpoint | Payload | Behavior |
|---|---|---|
| `POST /measurement-done` | `TestPayload` | Stores the result in `MeasurementStore`, keyed by `location.country_name` |
| `POST /vp-evaluated` | `EvalPayload` | Buffers the evaluation result in `EvalStore` for the orchestrator to process |

**Constructor parameters:**
- `store: MeasurementStore` — required, for measurement results
- `eval_store: EvalStore` — optional, for VP evaluation results (falls back to logging if not provided)

### `funneler.py` — `HyperQuackAPI`

Main interface to the Go Hyperquack service. Manages vantage point registration, job scheduling, result retrieval, and an internal `MeasurementAggregator`.

**Key methods:**

| Method | Purpose |
|---|---|
| `schedule_measurements(vps, services, targets, country)` | Registers VPs with Go, creates cross-product of (VP, target) jobs, sends to Go |
| `drain_raw_results(country)` | Returns raw `TestPayload` list for a country (used by orchestrator for VP health checks) |
| `try_get_results(country)` | Legacy method that parses raw results directly into `MeasurementResponse` list |
| `update_aggregator_vps(country, vps)` | Updates the aggregator's expected VP set for a country |
| `update_vps(new_vps, services)` | Registers new VPs with the Go server (deduplicates) |

**Debug mode:**

When `debug=True`, the Go server is not contacted. Instead:
- `_inject_debug_results()` generates synthetic `TestPayload`s with configurable block probability and delay.
- `_inject_debug_eval_results()` generates synthetic `EvalPayload`s with configurable success probability.

---

## Data Flow

```
Orchestrator
    │
    ├── schedule_measurements() ──► Go Server (/add-work)
    │
    └── drain_raw_results()  ◄──── MeasurementStore ◄── Server (/measurement-done) ◄── Go Server
                                   EvalStore         ◄── Server (/vp-evaluated)     ◄── Go Server
```

---
