---
phase: 02-state-persistence
plan: 01
subsystem: infra
tags: [networkx, graphml, persistence, atomic-write, deepcopy, json, tdd]

# Dependency graph
requires:
  - phase: 01-continuous-operation
    provides: RegionalNode soft_reset, BatchUCB scalar fields, MeasurementAggregator
provides:
  - "ActionSpaceBase.checkpoint_save() — mutation-safe GraphML snapshot"
  - "Infrastructure.utils.persistence — find_latest_state, load_graph_and_sidecar, validate_sidecar"
  - "BOOL_ATTRS constant for D-11 string-to-bool round-trip fix"
  - "REQUIRED_SIDECAR_KEYS constant for sidecar schema validation"
  - "Atomic write pattern (.tmp + os.replace) reusable across infra"
affects: [02-state-persistence/02-02, 02-state-persistence/02-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mutation-safe checkpoint via copy.deepcopy (no live-graph side effects)"
    - "POSIX-atomic file write via .tmp + os.replace"
    - "Boolean round-trip guard on GraphML load (cast str -> bool)"
    - "Required-key schema validation for JSON sidecars"

key-files:
  created:
    - "Infrastructure/utils/persistence.py"
    - "tests/infrastructure/test_persistence.py"
  modified:
    - "models/base/action_space.py"

key-decisions:
  - "checkpoint_save lives on ActionSpaceBase (graph owner), keeps PARENTS-list invariant on the live graph"
  - "Atomic write via .tmp + os.replace (T-02-02 mitigation) — same pattern reused in node sidecar writes (Plan 02)"
  - "Sidecar schema check is required-key only (no type check) — keeps validation cheap; type errors surface at restore time"
  - "BOOL_ATTRS uses lower-case attribute names because that is what GraphML serializes; matches PARENTS = 'parents' constant casing"

patterns-established:
  - "checkpoint_save(path) returns bool — caller controls path; instance has no opinion on file layout"
  - "load_graph_and_sidecar returns (None, None) on any failure — no exceptions propagate to callers, supports D-12 'start fresh on corruption'"
  - "find_latest_state requires both graphml AND json files — graphml without sidecar treated as no state"

requirements-completed: [PERS-04]

# Metrics
duration: ~5min
completed: 2026-05-03
---

# Phase 02 Plan 01: Persistence Primitives Summary

**Mutation-safe GraphML checkpoint method on ActionSpaceBase plus a stdlib-only persistence utility module exporting find_latest_state, load_graph_and_sidecar (with D-11 bool round-trip and D-12 corruption tolerance), and validate_sidecar — all backed by 21 passing TDD tests.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-05-03T17:25:00Z
- **Completed:** 2026-05-03T17:30:50Z
- **Tasks:** 2 (both TDD)
- **Files created:** 2 (persistence.py, test_persistence.py)
- **Files modified:** 1 (action_space.py)

## Accomplishments

- `ActionSpaceBase.checkpoint_save(path: str) -> bool` writes a GraphML snapshot via `copy.deepcopy`, reconnects orphaned sleeping nodes on the copy, converts PARENTS lists to "" on the copy only, and writes atomically via `.tmp` + `os.replace`. The live `self._graph` (node count, edge count, PARENTS list values) is provably untouched — verified by 4 dedicated mutation-detection tests.
- `Infrastructure/utils/persistence.py` exports three primitives needed by Plans 02 and 03: `find_latest_state` (rolling-checkpoint preferred over dated daily saves; returns `(None, None)` when sidecar missing), `load_graph_and_sidecar` (applies D-11 string-to-bool cast on `sleeping`/`explored`/`is_target_node`, restores PARENTS from "" to [], swallows all exceptions and returns `(None, None)` per D-12), `validate_sidecar` (REQUIRED_SIDECAR_KEYS schema check mitigating T-02-01).
- 21 unit tests covering checkpoint_save mutation-safety, atomic-write semantics, orphan reconnection, find_latest_state preference order and missing-sidecar fallbacks, load_graph_and_sidecar round-trip and corruption tolerance, and validate_sidecar key checks. Full infra suite (68 tests) green — no regressions in node, orchestrator, aggregator, or funneler tests.

## Task Commits

Each task was committed atomically following the TDD red-green cycle:

1. **Task 1 + 2 RED: failing tests for checkpoint_save and persistence module** — `19929df` (test)
2. **Task 1 GREEN: ActionSpaceBase.checkpoint_save with deepcopy and atomic write** — `5bbb912` (feat)
3. **Task 2 GREEN: Infrastructure.utils.persistence module** — `c5bdab7` (feat)

No refactor commits were needed — both implementations were minimal and focused on passing the tests without dead code or over-engineering.

_Note: A single RED commit covered both tasks because the tests share one file (`tests/infrastructure/test_persistence.py`); committing the test file twice would have caused a no-op or bogus second commit._

## Files Created/Modified

- `models/base/action_space.py` — added `import copy` (line 2) and `checkpoint_save()` method (lines 288-313) immediately after the existing `save()`. No changes to `save()` or any other existing method.
- `Infrastructure/utils/persistence.py` — new 135-line module: `BOOL_ATTRS` and `REQUIRED_SIDECAR_KEYS` constants plus the three public functions. Single `logger = logging.getLogger(__name__)` matching the aggregator/orchestrator pattern.
- `tests/infrastructure/test_persistence.py` — new 463-line test file with four test classes (TestCheckpointSave, TestFindLatestState, TestLoadGraphAndSidecar, TestValidateSidecar) and a `_make_minimal_action_space` helper that bypasses the heavy CSV-driven `build_graph()` via `unittest.mock.patch.object` so tests run in milliseconds.

## Decisions Made

- **Single RED commit covering both tasks' tests.** The plan declares both Task 1 and Task 2 as TDD with `tdd="true"`, but the natural test layout is one file (`test_persistence.py`) with separate test classes per primitive. Splitting RED into two commits would either require splitting the test file artificially or producing a duplicate / no-op commit. I committed all RED tests once, then implemented and committed the two production modules separately as their respective GREEN gates. The TDD audit trail is intact: `19929df` (RED, all tests fail) → `5bbb912` (GREEN for Task 1, TestCheckpointSave passes, persistence tests still fail) → `c5bdab7` (GREEN for Task 2, all 21 tests pass).
- **Test helper builds a tiny handcrafted graph** instead of running the real CSV-to-graph pipeline. `ActionSpaceBase.__init__` calls `build_graph()` (which needs a real DataFrame and feature columns) and `save()` (which writes to disk). Both are patched in `_make_minimal_action_space` so the constructor returns an instance whose live graph is the handcrafted DiGraph. This keeps tests deterministic, fast (sub-second), and free of fixture file dependencies.
- **`os.replace` mocked via `wraps=os.replace`** in the atomic-write test so the test still exercises the real syscall while we get an introspectable call record. This avoids the brittleness of asserting on file-system side effects only and makes the atomicity contract explicit in the test.

## Deviations from Plan

None — plan executed exactly as written. The two acceptance-criteria checks (presence of `import copy`, `checkpoint_save` signature, `copy.deepcopy(self._graph)`, `os.replace(tmp_path, path)`, no `self.reconnect_to_parents` inside `checkpoint_save`, `n_data[PARENTS] = ""` inside `checkpoint_save`; presence of the three public functions, BOOL_ATTRS, REQUIRED_SIDECAR_KEYS with the named keys; existence of test file with ≥10 test methods) were all met without auto-fixes.

## Issues Encountered

None. The cenv conda environment had networkx 3.5 already installed (verified before starting Task 1). No dependency installs, no schema migrations, no auth gates.

## Threat Model Compliance

| Threat ID | Disposition | Status |
|-----------|-------------|--------|
| T-02-01 (sidecar tampering) | mitigate | `validate_sidecar()` checks REQUIRED_SIDECAR_KEYS and returns False on any missing key; covered by `TestValidateSidecar.test_returns_false_when_keys_missing` and `test_returns_false_for_empty_dict` |
| T-02-02 (corrupted checkpoint on crash mid-write) | mitigate | `checkpoint_save()` writes `.tmp` then `os.replace()`; covered by `TestCheckpointSave.test_atomic_write_uses_os_replace` and `test_atomic_write_no_tmp_left_behind` |
| T-02-03 (info disclosure on disk) | accept | No mitigation required; documented as accepted in plan |

No new security-relevant surface introduced beyond what the plan's threat model anticipated. No threat flags raised.

## User Setup Required

None — local file I/O only, no external services.

## Next Phase Readiness

- **Plan 02 (RegionalNode save_checkpoint / save_daily / load_checkpoint)** — ready. Plan 02 will:
  - Call `self.model.action_space.checkpoint_save(graphml_path)` — already exists.
  - Call `find_latest_state(state_dir)` and `load_graph_and_sidecar(graphml, json)` for startup load — already exists.
  - Call `validate_sidecar(sidecar)` before restoring scalars — already exists.
- **Plan 03 (orchestrator timer + signal-handler + soft-reset wiring)** — depends on Plan 02 (RegionalNode wrappers), not this plan directly. This plan is unblocked.
- The `BOOL_ATTRS` and `REQUIRED_SIDECAR_KEYS` constants are public — Plan 02's `_write_sidecar()` should write the same keys to keep round-trip honest.

## Self-Check: PASSED

Verified files and commits:

- `models/base/action_space.py` — FOUND (modified, contains `checkpoint_save` at line 288)
- `Infrastructure/utils/persistence.py` — FOUND (created, 135 lines, exports the three functions)
- `tests/infrastructure/test_persistence.py` — FOUND (created, 463 lines, 21 test methods, all passing)
- Commit `19929df` — FOUND in git log (test RED)
- Commit `5bbb912` — FOUND in git log (Task 1 GREEN)
- Commit `c5bdab7` — FOUND in git log (Task 2 GREEN)
- Verification: `conda run -n cenv python -m pytest tests/infrastructure/test_persistence.py -v` → 21 passed
- Verification: `hasattr(ActionSpaceBase, 'checkpoint_save')` → `True`
- Verification: `from Infrastructure.utils.persistence import find_latest_state, load_graph_and_sidecar, validate_sidecar` → `OK`
- Full infra suite: 68 passed, 0 failed (no regressions)

---
*Phase: 02-state-persistence*
*Plan: 01*
*Completed: 2026-05-03*
