---
phase: 03-instrumentation-and-csv-output
plan: 01
subsystem: infrastructure-aggregator
tags: [instrumentation, aggregator, dataclass, plumbing, phase3-foundation]
requires:
  - "Phase 1.1 register_targets snapshot semantics (preserved)"
  - "Phase 02.1 _inflight_times delayed-result logger contract (preserved)"
provides:
  - "MeasurementResponse.scheduled_at_monotonic (float | None)"
  - "MeasurementResponse.vp_count (int | None)"
  - "MeasurementAggregator._schedule_times: Dict[(country,target), float]"
  - "MeasurementAggregator.register_targets schedule_times kwarg"
affects:
  - "Plan 03-02 (measurements CSV writer): consumes the two new fields"
  - "Plan 03-03 (tick instrumentation): may add more orchestrator timings"
tech-stack:
  added: []
  patterns:
    - "Modern union syntax (float | None) consistent with Pending.observed_value"
    - "Optional kwarg with None default to preserve existing call sites"
    - "Pop-on-finalize for unbounded-state guard (parallels _pending/_target_expected)"
key-files:
  created: []
  modified:
    - Infrastructure/utils/structures.py
    - Infrastructure/utils/aggregator.py
    - Infrastructure/main/orchestrator.py
    - tests/infrastructure/test_aggregator.py
    - tests/infrastructure/test_orchestrator.py
decisions:
  - "Compute now = time.monotonic() once and reuse for both schedule_times and _inflight_times so latency derivation in Plan 03-02 matches the orchestrator's delayed-result logger (D-04)"
  - "vp_count = len(votes) (actual majority count after drop_vp) rather than the originally-expected count (D-05)"
  - "schedule_times kwarg defaults to None so existing aggregator tests and callers stay unchanged (backward-compatibility guarantee)"
  - "Update TestRegisterTargetsWiring.test_tick_calls_register_targets to use mock.ANY for the new kwarg rather than coupling to a specific monotonic value"
metrics:
  duration: "~3 min wall clock"
  tasks_completed: 2
  files_modified: 5
  files_created: 0
  tests_added: 5
  completed_date: "2026-05-07"
---

# Phase 3 Plan 1: Plumb scheduled_at_monotonic and vp_count through the aggregator — Summary

Widened `MeasurementResponse` with two optional instrumentation fields and threaded the per-target monotonic schedule timestamp from `Orchestrator.tick()` Step 5 through `MeasurementAggregator.register_targets` → `_schedule_times` → `_try_finalize`, so every finalized response carries `scheduled_at_monotonic` (latency source) and `vp_count` (actual majority size). No CSV is written in this plan — Plan 03-02 (measurements CSV writer) is the consumer.

## What Was Done

### Task 1 — Widen MeasurementResponse + aggregator passthrough (`c353987`)

- **`Infrastructure/utils/structures.py`** — Added two optional fields to `MeasurementResponse`:
  - `scheduled_at_monotonic: float | None = None` (D-04 latency source)
  - `vp_count: int | None = None` (D-05 actual majority count)
  Modern union syntax matches the existing `Pending.observed_value` precedent (line 15).
- **`Infrastructure/utils/aggregator.py`**:
  - `__init__`: added `self._schedule_times: Dict[Tuple, float] = {}` book-keeping store.
  - `register_targets`: widened with optional `schedule_times: Dict[str, float] | None = None` kwarg. When provided, populates `_schedule_times[(country, target)]` so the finalize step can attach it to the emitted response. When absent (existing tests, debug paths), behaves identically to before.
  - `_try_finalize`: added `sched_mono = self._schedule_times.pop(key, None)` and constructed `MeasurementResponse(..., scheduled_at_monotonic=sched_mono, vp_count=len(votes))`. Pop-on-finalize guarantees no orphan entries (parallels `_pending` / `_target_expected` lifecycle, mitigates T-03-03 unbounded-growth threat). The `drop_vp` path naturally produces `vp_count = surviving_voter_count` because `len(votes)` is computed from the (already-shrunk) `expected` snapshot.

### Task 2 — Wire orchestrator schedule_times + add tests (`9229b1b`)

- **`Infrastructure/main/orchestrator.py`** `tick()` Step 5:
  - Restructured to compute `now = time.monotonic()` ONCE, build `schedule_times = {t: now for t in targets}`, pass it via the new kwarg to `register_targets`, and continue storing the same value in `self._inflight_times[country][t]` so the delayed-result logger and per-country reset callback contracts are preserved.
  - The `register_targets` call moved to before `schedule_measurements` (it already was) and now consumes the precomputed `now` from the schedule_times dict construction.
- **`tests/infrastructure/test_aggregator.py`** — Appended two new test classes (5 tests total):
  - `TestVpCountPassthrough::test_vp_count_three_when_three_voters` — vp_count == 3 with three voters in majority.
  - `TestVpCountPassthrough::test_vp_count_two_after_drop_vp` — vp_count == 2 after drop_vp shrinks expected from 3→2 (surviving voter count, per D-05).
  - `TestScheduledAtMonotonicPassthrough::test_schedule_time_attached_when_provided` — provided 100.5 flows through to MeasurementResponse.
  - `TestScheduledAtMonotonicPassthrough::test_schedule_time_none_when_not_provided` — absent schedule_times → None on the response (default preservation).
  - `TestScheduledAtMonotonicPassthrough::test_schedule_times_pop_after_finalize` — `_schedule_times` empty after finalize (no orphan entries, T-03-03 mitigation).
- **`tests/infrastructure/test_orchestrator.py`** — Updated `TestRegisterTargetsWiring.test_tick_calls_register_targets` to allow the new `schedule_times=ANY` kwarg via `unittest.mock.ANY`, avoiding tight coupling to the specific monotonic value picked at scheduling.

## Verification Run

- `python -m pytest tests/infrastructure/test_aggregator.py -x -q` → **13 passed** (8 pre-existing + 5 new).
- `python -m pytest tests/infrastructure/ -q --deselect <pre-existing failure>` → **128 passed, 1 deselected**.
- All grep gates from the plan's `<verify>` blocks report the expected counts:
  - `structures.py scheduled_at_monotonic: float | None = None` → 1
  - `structures.py vp_count: int | None = None` → 1
  - `aggregator.py self._schedule_times` → 3
  - `aggregator.py schedule_times: Dict[str, float] | None = None` → 1
  - `aggregator.py vp_count=len(votes)` → 1
  - `aggregator.py scheduled_at_monotonic=sched_mono` → 1
  - `orchestrator.py schedule_times=schedule_times` → 1
  - `orchestrator.py self._inflight_times[country][t] = now` → 1
- Smoke test: `MeasurementResponse(target='x', blocked=False)` constructs and `print(m.scheduled_at_monotonic, m.vp_count)` → `None None` (positional/two-arg construction unchanged).

## Deviations from Plan

**One small deviation**:

### `[Rule 3 - Test compatibility] Updated test_tick_calls_register_targets to use mock.ANY for the new kwarg`

- **Found during:** Task 2 verification (`pytest tests/infrastructure/`).
- **Issue:** `TestRegisterTargetsWiring.test_tick_calls_register_targets` asserted the exact 3-arg positional signature; the new `schedule_times` kwarg made it fail.
- **Fix:** Per the plan's own acceptance criteria (line 362 of 03-01-PLAN.md, "update the assertion to include it … via mock.ANY for the schedule_times parameter"), updated the assertion to use `mock.ANY` so the test does not couple to the specific monotonic value picked at scheduling.
- **Files modified:** `tests/infrastructure/test_orchestrator.py`
- **Commit:** `9229b1b` (folded into Task 2)

This was anticipated by the plan, so arguably not a deviation at all — recorded for completeness.

## Pre-existing Failures (Not in Scope)

`tests/infrastructure/test_orchestrator.py::TestVPRemovalOnEvalFailure::test_no_remove_when_all_pass` fails on the pre-Plan-03-01 codebase (verified by `git stash` reproduction) and is **unrelated to this plan**. Logged to `.planning/phases/03-instrumentation-and-csv-output/deferred-items.md` for separate triage. The remainder of the test suite (128 tests) passes.

## Acceptance Criteria — Status

- [x] `MeasurementResponse(target="x", blocked=True)` (positional, two args) constructs successfully and produces `scheduled_at_monotonic=None`, `vp_count=None`. (Verified.)
- [x] `MeasurementResponse(target="x", blocked=True, scheduled_at_monotonic=12345.6, vp_count=3)` constructs and exposes both new fields. (Verified.)
- [x] `register_targets("US", ["a.com"], {"vp1", "vp2"})` (no schedule_times) does not raise and behaves identically to before. (8 pre-existing aggregator tests pass.)
- [x] `register_targets("US", ["a.com"], {"vp1", "vp2"}, schedule_times={"a.com": 100.0})` populates `agg._schedule_times[("US", "a.com")] == 100.0`. (Smoke-tested + `test_schedule_time_attached_when_provided`.)
- [x] After two record() calls completing the majority, the resulting MeasurementResponse has `scheduled_at_monotonic == 100.5` and `vp_count == 2` (verified by `test_schedule_time_attached_when_provided`).
- [x] After `drop_vp` causes finalization with vp1+vp2 surviving, `MeasurementResponse.vp_count == 2` — confirms drop-vp path correctness (verified by `test_vp_count_two_after_drop_vp`).
- [x] `agg._schedule_times` is empty after finalize (verified by `test_schedule_times_pop_after_finalize`).
- [x] Orchestrator `tick()` Step 5 calls `register_targets(..., schedule_times={t: now for t in targets})` where `now = time.monotonic()`, AND the same `now` is also written to `self._inflight_times[country][t]`.
- [x] All 5 new tests pass; all previously-passing tests continue to pass; 1 pre-existing failure documented as out-of-scope.

## Plan Boundary

This plan owns ONLY the dataclass widening, aggregator book-keeping, and orchestrator call-site change. **No CSV is written in this plan**; the per-measurement CSV writer is owned by Plan 03-02, which now consumes `MeasurementResponse.scheduled_at_monotonic` and `MeasurementResponse.vp_count` as already-present fields.

## Self-Check: PASSED

- File `Infrastructure/utils/structures.py` exists and contains the new fields. (FOUND)
- File `Infrastructure/utils/aggregator.py` exists and contains `self._schedule_times`, the widened `register_targets` signature, and `vp_count=len(votes)` / `scheduled_at_monotonic=sched_mono`. (FOUND)
- File `Infrastructure/main/orchestrator.py` contains `schedule_times=schedule_times`. (FOUND)
- File `tests/infrastructure/test_aggregator.py` contains the 5 new tests. (FOUND)
- File `tests/infrastructure/test_orchestrator.py` updated to allow the new kwarg via `mock.ANY`. (FOUND)
- Commit `c353987` (Task 1) — present in `git log`. (FOUND)
- Commit `9229b1b` (Task 2) — present in `git log`. (FOUND)
