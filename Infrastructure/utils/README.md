# utils/

Shared data structures, thread-safe stores, and aggregation logic used across the Infrastructure system.

---

## Modules

### `structures.py`

Core data structures shared throughout the system.

**Enums:**

| Enum | Values | Purpose |
|---|---|---|
| `NodeState` | `IDLE`, `READY`, `AWAITING_MEASUREMENTS`, `DONE` | Regional node lifecycle states |
| `BatchSizeMethod` | `CONSTANT_VAL`, `VARY_ON_SUCCESS`, `NESTED_RL` | How batch size is determined |
| `BatchSelectionMethod` | `TOP_K_FROM_ARM`, `UNIFORM_SPREAD`, `WEIGHTED_SPREAD` | How targets are selected from arms |
| `PropagationMethod` | `ON_RECEIPT`, `IN_ORDER` | When rewards are propagated to the model |

**Pydantic models (for API serialization):**

| Model | Fields | Used By |
|---|---|---|
| `TestPayload` | `vp`, `location`, `service`, `test_url`, `response`, `anomaly`, `controls_failed`, `stateful_block` | Go server -> `/measurement-done` |
| `EvalPayload` | `vp`, `service`, `response`, `issue`, `template` | Go server -> `/vp-evaluated` |
| `LocationData` | `country_name`, `country_code` | Nested in `TestPayload` |
| `TestResponseData` | `matches_template`, `start_time`, `end_time` | Nested in `TestPayload` |

**Dataclasses:**

| Class | Fields | Purpose |
|---|---|---|
| `MeasurementResponse` | `target`, `blocked` | Parsed/aggregated result delivered to the model |
| `Job` | `keyword`, `services`, `vantage_point_predicate`, `tag` | Work item sent to Go server |
| `Tag` | `tag`, output/eval files, mongo config | Metadata for job tagging |
| `Pending` | `node_id`, `arm_key`, `arm_seq`, `blocked`, `reward`, `observed_value` | In-flight measurement tracking within BatchUCB |

### `store.py` — `MeasurementStore`

Thread-safe per-country result buffer for measurement payloads arriving from the Go server.

**Key methods:**

| Method | Purpose |
|---|---|
| `record_result(request_id, result)` | Store a result (called by server thread) |
| `get_country_batch(country, clear=True)` | Return and optionally clear all results for a country (non-blocking) |
| `get_ready_batch(block, timeout)` | Return results for any country that has data (optionally blocking) |

Results are keyed by country name. The server thread writes via `record_result`, and the orchestrator thread reads via `get_country_batch`.

### `eval_store.py` — `EvalStore`

Thread-safe buffer for VP evaluation results, with VP-to-country mapping.

The Go server's `EvalPayload` only contains the VP IP address (no country). The orchestrator calls `register_vp(vp, country)` before sending a VP for evaluation so that incoming results can be resolved.

**Key methods:**

| Method | Purpose |
|---|---|
| `register_vp(vp, country)` | Record which country a VP belongs to |
| `get_country(vp)` | Look up the country for an incoming eval |
| `record(payload)` | Buffer an incoming `EvalPayload` |
| `drain()` | Return and clear all buffered results |

### `aggregator.py` — `MeasurementAggregator`

Per-target, per-VP result aggregation buffer with majority-vote finalization.

When multiple VPs are active for a country, each measurement target produces one `TestPayload` per VP. The aggregator buffers these and emits a single `MeasurementResponse` per target once **all** expected VPs have reported.

**Majority vote rule:** A target is considered `blocked` if strictly more than 50% of VPs report it blocked. Ties resolve to not-blocked.

**Key methods:**

| Method | Purpose |
|---|---|
| `set_expected_vps(country, vps)` | Define which VPs must report before finalization |
| `record(country, vp, target, blocked)` | Record one VP's result; auto-finalizes if complete |
| `get_ready(country)` | Return and clear all finalized `MeasurementResponse`s |
| `drop_vp(country, vp)` | Remove a VP from expected set; re-evaluates all pending entries |

`drop_vp` is critical for handling VP removal mid-flight: any pending targets that were waiting on the removed VP are re-evaluated with the reduced expected set.

### `test_aggregator.py`

Unit tests for `MeasurementAggregator` and `EvalStore`. Run with:
```
python -m unittest Infrastructure.utils.test_aggregator -v
```

---

## Thread Safety

All stores and the aggregator use `threading.Lock` (or `threading.RLock` in the case of `VantagePoints`). The typical threading model is:

- **Server thread** (FastAPI/uvicorn): writes to `MeasurementStore` and `EvalStore`
- **Orchestrator thread** (main loop): reads from stores, writes to aggregator, reads from aggregator

No component requires external synchronization beyond its internal locks.

---
