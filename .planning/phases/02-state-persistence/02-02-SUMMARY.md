---
phase: 02-state-persistence
plan: 02
subsystem: infra
tags: [persistence, graphml, json, checkpoint, atomic-write, graph-merge, tdd]

# Dependency graph
requires:
  - phase: 02-state-persistence
    plan: 01
    provides: ActionSpaceBase.checkpoint_save, Infrastructure.utils.persistence (find_latest_state, load_graph_and_sidecar, validate_sidecar, REQUIRED_SIDECAR_KEYS)
provides:
  - "RegionalNode.save_checkpoint(state_dir, prefix=\"checkpoint\") — rolling checkpoint write"
  - "RegionalNode.save_daily(date_str, state_dir) — date-stamped daily snapshot"
  - "RegionalNode.load_checkpoint(state_dir) -> bool — auto-detect and merge saved state"
  - "RegionalNode._write_sidecar(json_path) — atomic JSON sidecar with all REQUIRED_SIDECAR_KEYS + metadata"
  - "initialize() auto-loads state on startup (D-09)"
affects: [02-state-persistence/02-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Atomic JSON write via .tmp + os.replace (mirrors checkpoint_save pattern)"
    - "Enum.name string serialization with class-name[name] restore"
    - "Graph merge via per-node attribute patching onto freshly built CSV graph (D-10)"
    - "Auto-detect saved state in initialize() — zero CLI flag (D-09)"

key-files:
  created: []
  modified:
    - "Infrastructure/main/node.py"
    - "tests/infrastructure/test_node.py"

key-decisions:
  - "load_checkpoint runs AFTER set_action_space() and patches Q-values onto the live graph rather than replacing it — preserves edge structure built from current CSV (matches RESEARCH.md graph-merge guidance)"
  - "Hyperparameter mismatch is a WARNING, never a hard failure — D-14 requires log-and-continue so a config drift never blocks resume"
  - "Each enum restoration is its own try/except KeyError block — preserves the literal `BatchSelectionMethod[` / `BatchSizeMethod[` / `PropagationMethod[` strings demanded by acceptance criteria and gives per-field warnings on bad sidecar names (T-02-04 mitigation)"
  - "save_checkpoint and save_daily wrap the graphml + sidecar write pair in a single try/except — one failure mode (logged) per save call, no half-written checkpoint state"

patterns-established:
  - "RegionalNode persistence methods take Path objects (state_dir, json_path) — caller owns layout decisions; instance has no opinion on file paths beyond filename conventions"
  - "Daily files use prefix action_space_<date>.graphml + state_<date>.json; rolling files use checkpoint.graphml + checkpoint.json — matches Plan 01 find_latest_state preference order"
  - "Graph merge silently drops saved-only nodes (CSV-removed targets) and silently keeps live-only nodes at default values (CSV-added targets) — no warnings, since CSV drift is expected in continuous operation"

requirements-completed: [PERS-01, PERS-02, PERS-03]

# Metrics
duration: ~5min
completed: 2026-05-03
---

# Phase 02 Plan 02: RegionalNode Save/Load API Summary

**Adds save_checkpoint, save_daily, _write_sidecar, and load_checkpoint to RegionalNode plus initialize() auto-load — wires Plan 01 primitives (ActionSpaceBase.checkpoint_save, find_latest_state, load_graph_and_sidecar, validate_sidecar) into the per-country lifecycle and patches saved Q-values onto a freshly built CSV graph (D-10), all backed by 13 new TDD tests with the full infra suite at 81/81 green.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-05-03T17:34:41Z
- **Completed:** 2026-05-03T17:39:36Z
- **Tasks:** 2 (both TDD)
- **Files created:** 0
- **Files modified:** 2 (node.py, test_node.py)

## Accomplishments

- `RegionalNode._write_sidecar(json_path)` writes the D-14 sidecar atomically: UCB counters (`exploration_epoch_num`, `current_epoch_num`), daily progress (`measurements_completed_today`), run config snapshot (`c`, `step_size`, `initial_value_estimate`, `selection_method.name`, `size_method.name`, `prop_method.name`), and metadata (`save_timestamp` UTC ISO-8601, `country_code`). Atomicity via `.tmp` + `os.replace` mirrors `checkpoint_save` in Plan 01 (T-02-02 mitigation).
- `RegionalNode.save_checkpoint(state_dir, prefix="checkpoint")` writes rolling `<state_dir>/<prefix>.graphml` and `<state_dir>/<prefix>.json`. Existing files are overwritten (D-02). All exceptions are caught and logged; the orchestrator never crashes on a checkpoint failure (D-12).
- `RegionalNode.save_daily(date_str, state_dir)` writes `action_space_<date>.graphml` + `state_<date>.json` (D-06). Per Pitfall 4 in RESEARCH.md, this must be called BEFORE `soft_reset()` so the captured `current_epoch_num` reflects the actual day's measurement count and `SLEEPING` flags reflect end-of-day state.
- `RegionalNode.load_checkpoint(state_dir)` returns True on a successful load: calls `find_latest_state` for `(graphml, json)` discovery, `load_graph_and_sidecar` for parsing (D-11 bool round-trip + D-12 corruption tolerance applied inside Plan 01's primitive), and `validate_sidecar` against `REQUIRED_SIDECAR_KEYS` (T-02-04 mitigation). On any failure path it returns False — caller can treat as "start fresh."
- Graph merge (D-10) patches `Q_VALUE`, `ACTION_ATTEMPTS`, `SLEEPING`, and `EXPLORED` from the saved graph onto matching nodes of the live graph (which was just built fresh from CSV in `set_action_space()`). Saved-only nodes are silently skipped (CSV-removed targets); live-only nodes keep `build_graph` defaults (CSV-added targets). The merge preserves the live graph's edge structure (which reflects the current CSV) and only swaps numeric/boolean leaf attributes.
- UCB scalar restore: `exploration_epoch_num`, `current_epoch_num` are copied directly. `c`, `step_size`, `initial_value_estimate` are compared and any mismatch logs a per-field WARNING (D-14: warn-don't-fail). Enum fields are restored individually via `BatchSelectionMethod[name]` / `BatchSizeMethod[name]` / `PropagationMethod[name]` indexing — each in its own try/except so an unknown enum name in a tampered sidecar surfaces as a single warning rather than aborting the entire load.
- `initialize()` extended to auto-detect state on startup: after building the fresh action space, it checks `Path(self.model.output_directory) / "state"` and calls `load_checkpoint(state_dir)` if the directory exists. No CLI flag (D-09); silent fall-through to fresh start when no state files exist or load fails.
- 13 new TDD tests across two test classes (`TestSaveCheckpoint` x7, `TestLoadCheckpoint` x6); full `tests/infrastructure/test_node.py` is 22/22 green; full infra suite (81 tests) is 81/81 green — zero regressions in aggregator, funneler, orchestrator, or persistence test files.

## Task Commits

Each task was committed atomically following the TDD red-green cycle:

1. **Task 1 RED: failing tests for save_checkpoint, save_daily, _write_sidecar** — `afc4704` (test)
2. **Task 1 GREEN: save_checkpoint, save_daily, _write_sidecar on RegionalNode** — `309ab64` (feat)
3. **Task 2 RED: failing tests for load_checkpoint** — `9ce8790` (test)
4. **Task 2 GREEN: load_checkpoint with graph merge + initialize() auto-load** — `c825f7e` (feat)

No refactor commits were needed — both implementations were minimal and matched the plan's `<action>` block + RESEARCH.md patterns 1:1.

## Files Created/Modified

- `Infrastructure/main/node.py` — added imports (`json`, `datetime/timezone`, persistence module), three save methods (`_write_sidecar` lines 138-167, `save_checkpoint` lines 169-184, `save_daily` lines 186-200), one load method (`load_checkpoint` lines 202-282), and a 5-line block in `initialize()` for auto-detect on startup. Final file: 421 lines.
- `tests/infrastructure/test_node.py` — added imports (`json`, `os`, `Path`, persistence enums), two test classes: `TestSaveCheckpoint` (7 tests covering file creation, overwrite, date-stamping, sidecar required keys, enum-name strings, atomic write, error isolation) and `TestLoadCheckpoint` (6 tests covering True/False returns, graph-merge invariants, hyperparameter-mismatch warning, enum restoration). Helpers: `_make_node` for save tests builds a mocked-model node; `_make_node` for load tests attaches a real `nx.DiGraph` so the merge has something to write into; `_write_saved_state` synthesizes a valid GraphML + JSON pair on disk for the load happy-path tests. Final file: 504 lines.

## Decisions Made

- **Per-enum literal try/except instead of a table-driven loop.** Acceptance criterion required the literal string `BatchSelectionMethod[` to appear in `node.py`. A table-driven loop (`for field, cls in [...]`) restores the enums correctly but obscures the type names from a static grep. I unrolled the three restorations into individual try/except blocks — each is one literal call to `EnumClass[sidecar[key]]`. This satisfies the acceptance criterion exactly while keeping per-field warning messages (so a malformed sidecar entry for `prop_method` doesn't blame `selection_method`).
- **Hyperparameter mismatch comparison loops over a tuple of three field names.** Cleaner than three explicit `if`s because the warning message format is identical across `c`, `step_size`, and `initial_value_estimate`. The loop reads `getattr(self.model, hp, None)` so a mocked model in tests doesn't blow up if a hyperparameter is missing.
- **Graph merge skips saved-only nodes silently.** D-10 requires that targets removed from the CSV are dropped. The natural implementation (don't add them to the live graph) needs no positive action — they're already absent because we built fresh. So the merge loop iterates `saved_graph.nodes()` and skips any `node_id` not in `live_graph` via `continue`, no warning logged. This is the expected behavior in continuous operation, not an error condition.
- **Auto-load runs at the end of `initialize()` rather than replacing `set_action_space()`.** The plan considered two architectures: (a) load a saved graph and use it as the live graph, or (b) build fresh and patch saved values onto it. Option (b) wins because the freshly-built graph reflects current CSV edge structure (which may have new arms or different feature partitions); patching only Q-values and counts onto it preserves both the latest topology and the learning history. This matches RESEARCH.md Pattern 4 ("rebuild then patch") and the plan's note in Task 2's `<action>`.
- **Tests use real `nx.DiGraph` in the live action space and a real disk-written GraphML in the saved state.** The point of `load_checkpoint` is to round-trip a serialized graph through `nx.read_graphml`; mocking the graph merge would test nothing. The `_make_node` helper for `TestLoadCheckpoint` builds a real `nx.DiGraph` with the same `create_default_node_attributes` used in production, and `_write_saved_state` calls `nx.write_graphml` on a synthetic saved graph. Both `find_latest_state` and `load_graph_and_sidecar` from Plan 01 are exercised end-to-end.

## Deviations from Plan

None — plan executed exactly as written.

The two acceptance-criteria checklists (Task 1: 9 items, Task 2: 9 items) are 18/18 PASS. The three verification commands all succeed:

```
$ conda run -n cenv python -m pytest tests/infrastructure/test_node.py -x -v
...
22 passed in 0.41s

$ conda run -n cenv python -m pytest tests/infrastructure/
...
81 passed in 3.31s

$ conda run -n cenv python -c "from Infrastructure.main.node import RegionalNode; \
    print(hasattr(RegionalNode, 'save_checkpoint'), hasattr(RegionalNode, 'save_daily'), \
    hasattr(RegionalNode, 'load_checkpoint'))"
True True True
```

## Issues Encountered

None. The cenv conda environment, networkx 3.5, and Plan 01's `Infrastructure.utils.persistence` module were all already in place. No dependency installs, no import path issues, no test fixture surprises.

The only minor course-correction was the enum-restoration code style: my first pass used a table-driven loop (`for field, cls in [...]`) which functionally satisfied test 6 of TestLoadCheckpoint but caused the `BatchSelectionMethod[` literal-string acceptance check to fail. I unrolled the loop into three explicit try/except blocks, re-ran tests (still 22/22 green), and re-checked the criteria. Total iteration: one Edit + one test run.

## Threat Model Compliance

| Threat ID | Disposition | Status |
|-----------|-------------|--------|
| T-02-04 (sidecar tampering on restore) | mitigate | `validate_sidecar` runs before any scalar restore (covered by `test_load_checkpoint_returns_false_on_corrupt_files` via the `load_graph_and_sidecar` (None, None) path, which short-circuits before validation; and by Plan 01's `TestValidateSidecar` for the missing-keys path). Enum restoration wrapped in per-field try/except KeyError (covered behaviorally by `test_load_checkpoint_restores_enum_fields` — valid path; the tamper path is handled by code coverage of the except branches). |
| T-02-05 (graph-merge elevation of privilege) | accept | Graph merge only copies four numeric/boolean attributes (`Q_VALUE`, `ACTION_ATTEMPTS`, `SLEEPING`, `EXPLORED`) between graph objects. No code path is invoked. Documented as accepted in plan. |

No new security-relevant surface introduced beyond what the plan's threat model anticipated. No threat flags raised.

## User Setup Required

None — local file I/O only, no external services, no new dependencies.

## Next Phase Readiness

- **Plan 03 (orchestrator timer + signal handler + soft-reset wiring)** — ready. Plan 03 will:
  - Call `node.save_checkpoint(state_dir)` from a `_checkpoint_all_nodes()` method gated by a monotonic timer in `run_forever()` — already exists.
  - Call `node.save_daily(date_str, state_dir)` from `_perform_soft_reset()` BEFORE `node.soft_reset()` — already exists.
  - Call `node.save_checkpoint(state_dir)` from `_shutdown()` after `node.write_stats()` — already exists.
  - Auto-load on startup is already wired into `initialize()` via `self.load_checkpoint(state_dir)` — Plan 03 needs no extra startup hook.
- The orchestrator only needs a `_state_dir_for(country)` helper (one-liner: `Path(self.output_folder) / country.replace(" ", "_") / "state"`) to bridge per-country path construction to the per-node save calls. All node-side persistence semantics — atomicity, error isolation, sidecar schema, graph merge — are handled by this plan.

## Self-Check: PASSED

Verified files and commits:

- `Infrastructure/main/node.py` — FOUND (modified, 421 lines, contains `save_checkpoint`, `save_daily`, `_write_sidecar`, `load_checkpoint`, and `state_dir = Path(self.model.output_directory) / "state"` in `initialize()`).
- `tests/infrastructure/test_node.py` — FOUND (modified, 504 lines, contains `TestSaveCheckpoint` and `TestLoadCheckpoint` classes with 13 new test methods).
- Commit `afc4704` — FOUND in git log (Task 1 RED).
- Commit `309ab64` — FOUND in git log (Task 1 GREEN).
- Commit `9ce8790` — FOUND in git log (Task 2 RED).
- Commit `c825f7e` — FOUND in git log (Task 2 GREEN).
- Verification: `conda run -n cenv python -m pytest tests/infrastructure/test_node.py -v` → 22 passed.
- Verification: `conda run -n cenv python -m pytest tests/infrastructure/` → 81 passed (no regressions in aggregator, funneler, orchestrator, or persistence).
- Verification: `hasattr(RegionalNode, 'save_checkpoint') and hasattr(RegionalNode, 'save_daily') and hasattr(RegionalNode, 'load_checkpoint')` → True.

---
*Phase: 02-state-persistence*
*Plan: 02*
*Completed: 2026-05-03*
