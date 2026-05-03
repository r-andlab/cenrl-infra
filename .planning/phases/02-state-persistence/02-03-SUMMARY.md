---
phase: 02-state-persistence
plan: 03
subsystem: infra
tags: [orchestrator, persistence, checkpoint, signal-handler, soft-reset, monotonic-timer]

# Dependency graph
requires:
  - phase: 02-state-persistence
    plan: 02
    provides: RegionalNode.save_checkpoint, RegionalNode.save_daily, RegionalNode.load_checkpoint, initialize() auto-load
  - phase: 02-state-persistence
    plan: 01
    provides: ActionSpaceBase.checkpoint_save, Infrastructure.utils.persistence
provides:
  - "Orchestrator._state_dir_for(country) — per-country state path helper"
  - "Orchestrator._checkpoint_all_nodes() — fan-out hourly checkpoint with per-node error isolation"
  - "Hourly monotonic checkpoint timer wired into run_forever()"
  - "Daily save_daily() in _perform_soft_reset() ordered BEFORE soft_reset() (Pitfall 4)"
  - "Shutdown save_checkpoint() in _shutdown() after write_stats() (D-03)"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Monotonic-time hourly trigger in run_forever() mirroring the _check_delayed_results() pattern"
    - "Per-node try/except isolation around save calls (T-02-06, T-02-07 mitigations)"
    - "Save-before-reset ordering enforced by both production code and a dedicated test"

key-files:
  created: []
  modified:
    - "Infrastructure/main/orchestrator.py"
    - "tests/infrastructure/test_orchestrator.py"

key-decisions:
  - "TDD task ordered after the production wiring (Task 1 first) because the orchestrator must compile before any test in test_orchestrator.py can import Orchestrator. The new tests behaviorally validate the wiring; nothing in Task 2 changes production code."
  - "Updated existing test fixtures (TestRunForeverLoopCondition._make_orchestrator and TestDailyResetTrigger._make_orchestrator) to add _last_checkpoint_time, _checkpoint_interval, and output_folder so the new attributes referenced by run_forever()/_perform_soft_reset() do not crash legacy tests. This is a Rule 3 fix (blocking issue caused by the production change)."
  - "Hourly checkpoint check placed AFTER _check_delayed_results() and BEFORE tick() in run_forever(), matching the plan's <action> step 4 exactly. Variable named now_mono to avoid shadowing the now used in tick()'s scheduling code."
  - "_perform_soft_reset wraps save_daily in try/except but soft_reset() is unconditional — a failing daily save logs and continues so the daily reset still completes (T-02-07 mitigation)."
  - "_shutdown adds a SECOND try/except block for save_checkpoint AFTER the existing write_stats try/except — so a failing write_stats does not skip the save, and a failing save does not skip subsequent nodes."

patterns-established:
  - "Per-country state paths: <output_folder>/<country with spaces -> underscores>/state — owned by orchestrator helper, consumed by node save methods"
  - "Three-trigger save model is fully wired: hourly (timer), daily (boundary), shutdown (signal). Plans 01 and 02 primitives are now reachable from production code paths."

requirements-completed: [PERS-01, PERS-02, PERS-03]

# Metrics
duration: ~4min
completed: 2026-05-03
---

# Phase 02 Plan 03: Orchestrator Save Trigger Wiring Summary

**Wires the three save triggers (hourly checkpoint timer, daily-boundary save_daily before soft_reset, shutdown save_checkpoint after write_stats) into the orchestrator and adds 14 unit tests across 5 new test classes — completing Phase 02 state persistence with the full infra suite at 95/95 green.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-05-03T17:43:55Z
- **Completed:** 2026-05-03T17:47:21Z
- **Tasks:** 2 (one auto, one TDD)
- **Files created:** 0
- **Files modified:** 2 (orchestrator.py, test_orchestrator.py)

## Accomplishments

- `Orchestrator.__init__` extended with two new instance variables — `_last_checkpoint_time: float = time.monotonic()` and `_checkpoint_interval: float = 3600.0` — placed immediately after the existing `_inflight_times` block. No new imports needed; `time`, `Path`, `datetime`, and `timezone` were already available.
- `Orchestrator._state_dir_for(country: str) -> Path` returns `Path(self.output_folder) / country.replace(" ", "_") / "state"`. Mirrors the existing per-country path construction used in `RegionalNode.__init__` so saved files land alongside the country's CSV output.
- `Orchestrator._checkpoint_all_nodes()` iterates `self.agents.items()` and calls `node.save_checkpoint(state_dir)` for each, wrapped in per-node try/except. A failing node logs an error and the loop continues (T-02-06 mitigation).
- `run_forever()` now contains a monotonic-clock hourly check between `_check_delayed_results()` and `tick()`: `if now_mono - self._last_checkpoint_time >= self._checkpoint_interval: self._checkpoint_all_nodes(); self._last_checkpoint_time = now_mono`. The variable name `now_mono` avoids shadowing the `now` used inside `tick()`'s scheduling code.
- `_shutdown()` extended with a second try/except block (after the existing `write_stats()` block) that calls `node.save_checkpoint(state_dir)` per country. Graceful shutdown now persists state for every active node — D-03/D-04 wiring complete.
- `_perform_soft_reset()` rewritten to compute `date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")` once, then iterate agents calling `node.save_daily(date_str, state_dir)` BEFORE `node.soft_reset()`. The ordering is the Pitfall 4 mitigation: `soft_reset()` zeroes `current_epoch_num` and clears SLEEPING via `wake_up_all_nodes()`, so the daily snapshot must be captured first.
- 14 new tests across 5 new classes (`TestStateDirFor`, `TestCheckpointAllNodes`, `TestHourlyCheckpointTimer`, `TestShutdownSavesCheckpoint`, `TestDailySaveBeforeReset`) covering all 9 plan-specified behaviors plus 5 additional cases (path return type, `Path` instance check, error isolation paths, write_stats-before-save_checkpoint ordering, soft_reset-still-runs-when-save-fails). Full `tests/infrastructure/test_orchestrator.py` is 36/36 green; full infra suite (95 tests) is 95/95 green — no regressions in node, persistence, aggregator, or funneler tests.

## Task Commits

1. **Task 1: orchestrator wiring (timer vars + helper + checkpoint method + extended _shutdown/_perform_soft_reset)** — `d2e3809` (feat). Also updated 2 existing test fixtures to set `_last_checkpoint_time`, `_checkpoint_interval`, `output_folder` so legacy `run_forever()`/`_perform_soft_reset()` tests still run after the new instance variables and method calls were added.
2. **Task 2: 14 new orchestrator save-trigger tests across 5 classes** — `bb8a935` (test).

No refactor commits were needed.

## Files Created/Modified

- `Infrastructure/main/orchestrator.py` — modified, 646 lines (was 587). Added 4 new methods/sections: instance vars in `__init__` (lines 105-108), `_state_dir_for` (lines 357-363), `_checkpoint_all_nodes` (lines 365-376), rewritten `_perform_soft_reset` (lines 379-401), extended `_shutdown` (saves added at lines 317-321), hourly timer in `run_forever` (lines 274-277).
- `tests/infrastructure/test_orchestrator.py` — modified, 727 lines (was 497). Added imports (`Path`, `call`), updated 2 existing fixtures, appended 5 new test classes containing 14 test methods.

## Decisions Made

- **TDD ordering reversed: production wiring first, then tests.** The plan declared Task 2 as `tdd="true"` but the orchestrator must compile and import without `AttributeError` before *any* test in `test_orchestrator.py` can run — adding new instance vars (`_last_checkpoint_time`, etc.) to `run_forever()` without also adding them to existing test fixtures would have broken the file's import or made every existing test red. I implemented Task 1 (production + fixture updates) as a single feat commit, then Task 2 (new tests) as a single test commit. This satisfies the plan's verification criteria (all tests pass) while preserving the test-as-validator role. Documented under deviations.
- **Existing test fixtures updated alongside production code.** Three legacy tests (`TestRunForeverLoopCondition`, `TestDailyResetTrigger`) bypass `__init__` and assemble Orchestrator instance state by hand. Adding new attributes referenced by `run_forever()` required adding `_last_checkpoint_time`, `_checkpoint_interval`, and `output_folder` to those fixture builders. This is Rule 3 (auto-fix blocking issue).
- **Variable name `now_mono` for the hourly timer comparison.** The plan's `<action>` step 4 uses `now`, but `tick()` already has a local `now` for measurement scheduling. Renamed to `now_mono` to make the time domain explicit (monotonic clock, not wall clock) and to avoid any future shadowing accidents if these two methods are ever inlined.
- **Path-equality assertions use `Path("/tmp/out/Germany/state")` rather than string equality.** The plan's path helper returns `Path` objects; comparing with `Path(...)` rather than `str` makes platform-portable assertions and exercises the same equality semantics production code will see.
- **Per-enum / per-method MagicMock auto-handling.** `MagicMock` auto-creates `save_checkpoint`, `save_daily`, `soft_reset` methods on demand, so the test fixtures don't need to wire them explicitly. The ordering tests use `node.method_calls` to inspect call sequences — this is the standard idiom (`unittest.mock.NonCallableMock.method_calls`) for verifying call order without coupling to the underlying implementation.

## Deviations from Plan

### Rule 3 - Blocking issue: Existing test fixtures missing new instance attributes

**Found during:** Task 1 (after editing orchestrator.py, three pre-existing tests failed with `AttributeError: 'Orchestrator' object has no attribute '_last_checkpoint_time'`).

**Issue:** `TestRunForeverLoopCondition._make_orchestrator()` and `TestDailyResetTrigger._make_orchestrator()` use `Orchestrator.__new__(Orchestrator)` to bypass `__init__` and manually populate instance state. After adding `_last_checkpoint_time` and `_checkpoint_interval` to `__init__`, those fixtures lacked the new fields and `run_forever()` raised on the first iteration. Additionally, `TestDailyResetTrigger` fixtures lacked `output_folder`, which `_perform_soft_reset()` now needs via `_state_dir_for()`.

**Fix:** Added `_last_checkpoint_time = time.monotonic()`, `_checkpoint_interval = 3600.0`, and `output_folder = "/tmp/test_orchestrator_output"` to both `_make_orchestrator()` helpers. No changes to the test bodies were required.

**Files modified:** `tests/infrastructure/test_orchestrator.py`

**Commit:** `d2e3809` (folded into the Task 1 commit since the production change and the fixture update must land together to keep the suite green).

### TDD task ordering: production code landed in Task 1 before Task 2's RED phase

**Found during:** Reading the plan's Task 1 (auto) + Task 2 (TDD) structure.

**Issue:** Task 2 is declared `tdd="true"` with a behavior list mirroring the production code added in Task 1. A strict TDD cycle would require writing tests BEFORE the production code — but Task 1 must run first per the plan's task order, and writing the new tests against a not-yet-modified orchestrator would either need the tests stubbed out or would force an artificial sequencing that contradicts the plan.

**Fix:** Implemented Task 1's production wiring + the existing-test-fixture fixes as a `feat` commit (`d2e3809`), then committed Task 2's 14 new tests as a single `test` commit (`bb8a935`). Both ran green on commit. The TDD audit trail shows production code first, then comprehensive test coverage validating the wiring — diverging from RED→GREEN→REFACTOR in the strict sense but matching the plan's task ordering and acceptance criteria.

**Files modified:** none beyond the planned scope.

## Issues Encountered

The 3 pre-existing test fixture failures described above. Fixed inline as Rule 3 deviations; no other surprises. The cenv conda environment, `time.monotonic()`, `pathlib.Path`, and the Plan 02 RegionalNode methods (`save_checkpoint`, `save_daily`, `soft_reset`, `write_stats`) all worked as documented.

## Threat Model Compliance

| Threat ID | Disposition | Status |
|-----------|-------------|--------|
| T-02-06 (one node's save failure prevents others from checkpointing) | mitigate | `_checkpoint_all_nodes` and `_shutdown` both wrap `node.save_checkpoint()` in per-node try/except. Verified by `TestCheckpointAllNodes.test_continues_after_node_save_raises` and `TestShutdownSavesCheckpoint.test_shutdown_continues_when_one_node_save_raises`. |
| T-02-07 (save_daily failure blocks soft_reset) | mitigate | `_perform_soft_reset` wraps only the `save_daily` call in try/except; `soft_reset()` is invoked unconditionally. Verified by `TestDailySaveBeforeReset.test_soft_reset_runs_even_when_save_daily_raises`. |

No new security-relevant surface introduced beyond what the plan's threat model anticipated. No threat flags raised.

## User Setup Required

None — local file I/O only, no external services, no new dependencies, no schema migrations.

## Next Phase Readiness

- All three save triggers are wired and the full Phase 02 state-persistence feature is functional. With the orchestrator running:
  - **Hourly:** every 60 minutes, `_checkpoint_all_nodes()` writes `<output>/<country>/state/checkpoint.{graphml,json}` for each active node (D-01, D-02).
  - **Daily:** at the next midnight UTC reset, `_perform_soft_reset()` writes `action_space_<date>.graphml` + `state_<date>.json` BEFORE `soft_reset()` clears Q-values (D-06, Pitfall 4).
  - **Shutdown:** on SIGTERM/SIGINT, `_shutdown()` calls both `write_stats()` and `save_checkpoint()` per node (D-03).
- **Auto-load on startup** is wired in `RegionalNode.initialize()` (Plan 02 deliverable) — when a country gets its first good VP and a `RegionalNode` is constructed in `_process_eval_results()`, the constructor calls `initialize()` which checks `<output>/<country>/state/` and merges any saved Q-values onto the freshly built CSV graph.
- No further orchestrator changes are needed for Phase 02. Future phases can rely on per-country state directories existing under `<output_folder>/<Country>/state/` for any cross-day analysis tooling.

## Self-Check: PASSED

Verified files and commits:

- `Infrastructure/main/orchestrator.py` — FOUND (modified, 646 lines, contains `_state_dir_for`, `_checkpoint_all_nodes`, hourly timer in `run_forever`, `node.save_checkpoint` in `_shutdown`, `node.save_daily` BEFORE `node.soft_reset` in `_perform_soft_reset`)
- `tests/infrastructure/test_orchestrator.py` — FOUND (modified, 727 lines, contains 5 new test classes with 14 new test methods, full file 36/36 green)
- Commit `d2e3809` — FOUND in git log (Task 1 feat: orchestrator wiring)
- Commit `bb8a935` — FOUND in git log (Task 2 test: 14 new tests across 5 classes)
- Verification: `conda run -n cenv python -m pytest tests/infrastructure/ -x -v` → 95 passed
- Verification: `conda run -n cenv python -c "from Infrastructure.main.orchestrator import Orchestrator; o = object.__new__(Orchestrator); print(type(o._state_dir_for.__func__))"` → `<class 'function'>`
- Verification: grep confirms `node.save_daily(date_str, state_dir)` precedes `node.soft_reset()` in `_perform_soft_reset` (lines 395 vs 398)

---
*Phase: 02-state-persistence*
*Plan: 03*
*Completed: 2026-05-03*
