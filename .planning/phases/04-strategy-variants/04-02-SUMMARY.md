---
phase: 04-strategy-variants
plan: 02
subsystem: Infrastructure/weighted-vote-aggregation
tags: [aggregation, weighted-vote, beta-smoothing, vp-weights, wr-06-snapshot]
requires:
  - Infrastructure/utils/aggregator.py (post-Plan 01 state: aggregation_method kwarg + _vp_weights buffer)
  - Infrastructure/main/orchestrator.py (post-Plan 01 state: STRATEGY_MAP + cached enums + _vp_failure_counts)
  - Infrastructure/utils/structures.py (AggregationMethod enum from Plan 01)
provides:
  - Orchestrator._vp_outcomes: (country, vp) -> [success, fail] cumulative ledger
  - Orchestrator._beta_smoothed_weight(s, f) static helper
  - Orchestrator._compute_vp_weights(country, vps) per-tick weight builder
  - MeasurementAggregator.set_vp_weights(country, weights) push API
  - MeasurementAggregator._target_weights per-target weight snapshot
  - MeasurementAggregator._weighted_fallback_warned warn-once tracker
  - MeasurementAggregator._try_finalize enum-dispatched WEIGHTED_VOTE branch
affects:
  - Default-run aggregation behavior is unchanged (bit-for-bit; D-04)
  - --aggregation weighted now switches behavior measurably (STRT-01 SC1)
tech-stack:
  added: []
  patterns:
    - Beta(1,1) Laplace-smoothed posterior mean for binary outcomes
    - Per-target weight snapshot at scheduling time (WR-06 invariant extension)
    - Warn-once-per-key set-membership guard
    - Enum dispatch via `is` operator (Pattern 1 from 04-PATTERNS.md)
key-files:
  created: []
  modified:
    - Infrastructure/main/orchestrator.py
    - Infrastructure/utils/aggregator.py
    - tests/infrastructure/test_aggregator.py
    - tests/infrastructure/test_orchestrator.py
decisions:
  - D-01 (dispatch on self.aggregation_method inside _try_finalize)
  - D-02 (per-VP success-rate weights from extended _vp_outcomes ledger; pull v2 ADVS-02 forward; preserve history across rotations)
  - D-04 (defaults bit-for-bit unchanged; MAJORITY_VOTE path numerically equivalent to pre-Plan-01 line)
  - Claude's Discretion: zero-weight fallback to MAJORITY with warn-once-per-(country, target) at WARNING level
  - Open Question 1 (RESOLVED): new VP IP after rotation gets fresh (0, 0); rejected VP history is left in _vp_outcomes (never queried again because the IP is no longer in any active set)
metrics:
  duration_minutes: 6
  tasks_completed: 2
  files_modified: 4
  completed: 2026-05-11
---

# Phase 04 Plan 02: WEIGHTED_VOTE Aggregation Summary

Wave 2 of strategy-variants. Implements WEIGHTED_VOTE aggregation end-to-end: the orchestrator's existing `_vp_failure_counts` cluster is extended with a parallel `_vp_outcomes: (success, fail)` ledger that accumulates indefinitely; Beta(1,1) Laplace-smoothed weights are computed each scheduling round and pushed into the aggregator BEFORE `register_targets` so the per-target snapshot freezes them at schedule time (WR-06 invariant); `_try_finalize` branches on `aggregation_method` to a strict `weighted_sum > total_weight / 2` comparison with cold-start 0.5 default and zero-weight fallback to majority. This pulls v2 ADVS-02 (Confidence-weighted aggregation using VP reliability history) forward into Phase 4 by user design (D-02) — weights are real, not stubbed.

## What Changed

### Per-(country, vp) Outcomes Ledger (Task 1, Orchestrator)

- `Infrastructure/main/orchestrator.py` — added `self._vp_outcomes: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])` to `Orchestrator.__init__`, immediately after the existing `_vp_failure_counts` declaration. Two-element list (not tuple) for in-place index increments. The two trackers serve different purposes: `_vp_failure_counts` is reset on success (to control VP rejection at `VP_FAILURE_THRESHOLD`); `_vp_outcomes` accumulates forever (Beta smoothing requires cumulative counts; D-02 explicit: "needs to keep history rather than wiping").
- `_check_vp_health` — extended with two `+= 1` mutations: `_vp_outcomes[key][1] += 1` on `controls_failed=True` (immediately after the existing failure-counter increment), and `_vp_outcomes[key][0] += 1` on success (immediately after the existing `_vp_failure_counts.pop(key, None)`). The success-branch `pop` is preserved verbatim — failure-count reset semantics are unchanged.
- VP rejection path (`del self._vp_failure_counts[key]` at threshold): intentionally does NOT delete the corresponding `_vp_outcomes[key]` entry. Per Open Question 1 (RESOLVED in 04-RESEARCH.md), a new VP IP gets a fresh `(0, 0)` from `defaultdict` (cold-start 0.5 via Beta); the rejected VP's history is left in the ledger but never queried again because that IP is no longer in any country's active VP set. Memory cost is O(rejected VPs); acceptable for thesis-scale runs.

### Beta-Smoothed Weight Helpers (Task 1, Orchestrator)

- `@staticmethod _beta_smoothed_weight(success, fail)` — returns `(s + 1) / (s + f + 2)`. Cold-start `(0, 0) -> 0.5` (neutral); single-success `(1, 0) -> 2/3` (not the naive `1.0` trap); long-history converges to the true success rate; divide-by-zero impossible by construction. Standard Bayesian-vote pattern (RESEARCH Pitfall 2).
- `_compute_vp_weights(country, vps)` — builds `{vp_ip: weight}` for the country's active VP set by reading `self._vp_outcomes.get((country, vp), [0, 0])` and applying `_beta_smoothed_weight`. VPs with no recorded outcomes default to 0.5 via the `defaultdict`/`.get` fallback.

### Weight Push at Scheduling Time (Task 1, Orchestrator)

- `tick()` step 5 (scheduling) — added two new lines immediately before the existing `register_targets` call:
  ```python
  vp_weights = self._compute_vp_weights(country, set(active_vps))
  self.api.aggregator.set_vp_weights(country, vp_weights)
  ```
  Push happens BEFORE `register_targets` so the per-target snapshot (added in `register_targets` below) reads the just-pushed values. Cadence matches `register_targets` (Pitfall 9 / Assumption A4 / WR-06 invariant).

### Aggregator Per-Target Weight Snapshot (Task 1, Aggregator)

- `MeasurementAggregator.__init__` — added `self._target_weights: Dict[Tuple, Dict[str, float]] = {}` immediately after the Plan 01 `_vp_weights` line. Drained alongside `_target_expected`.
- `set_vp_weights(country, weights)` — new method between `set_expected_vps` and `register_targets`. Takes `self._lock` and stores `dict(weights)` (defensive copy) under `self._vp_weights[country]`.
- `register_targets` body — inside the per-target `for` loop, AFTER the WR-06 duplicate-skip guard, AFTER `self._target_expected[key] = snapshot`, snapshots `self._target_weights[key] = dict(self._vp_weights.get(country, {}))`. Empty dict if no weights pushed yet (cold-start handled at finalize via 0.5 default).
- `_try_finalize` cleanup — both the WR-01 zero-expected branch (after `del self._target_expected[key]`) and the normal finalize branch pop `self._target_weights.pop(key, None)`. No leak.

### WEIGHTED_VOTE Dispatch in _try_finalize (Task 2, Aggregator)

- `__init__` — added `self._weighted_fallback_warned: Set[Tuple[str, str]] = set()` warn-once tracker.
- `_try_finalize` — replaced the Plan 01 single-line `votes = [vp_map[v] for v in expected]; majority_blocked = sum(votes) > len(votes) / 2` with an enum-dispatched block:
  - `votes_pairs = [(v, vp_map[v]) for v in expected]` (carries vp_ip alongside vote for weight lookup).
  - `if self.aggregation_method is AggregationMethod.MAJORITY_VOTE:` — `sum(b for _, b in votes_pairs) > len(votes_pairs) / 2`. Numerically equivalent to the original line; default-run is bit-for-bit unchanged (D-04).
  - `elif self.aggregation_method is AggregationMethod.WEIGHTED_VOTE:` — reads the per-target snapshot via `target_weights = self._target_weights.get(key, {})`; VPs missing from the snapshot default to 0.5 (Beta cold-start). Computes `total_weight = sum(target_weights.get(vp, 0.5) for vp, _ in votes_pairs)`. If `total_weight == 0.0`, falls back to majority and adds key to `_weighted_fallback_warned` (warn-once log line). Otherwise computes `weighted_sum = sum(target_weights.get(vp, 0.5) * b for vp, b in votes_pairs)` and applies `weighted_sum > total_weight / 2` — strict-greater matches MAJORITY's no-block-on-tie semantics.
  - `else:` defensive branch logs error and falls back to majority (unreachable in practice because `argparse choices=` filters at parse time).
- `votes` list reassignment (`votes = [b for _, b in votes_pairs]`) preserves the downstream `vp_count=len(votes)` line on `MeasurementResponse` without modification.

### Test Coverage (Tasks 1 & 2)

`tests/infrastructure/test_aggregator.py` — 19 new tests added, 13 existing tests preserved (32 total, all passing):

- `TestSetVpWeights` (4 tests): stores per-country dict, overwrites previous, per-country independence, defensive copy.
- `TestPerTargetWeightSnapshot` (5 tests): WR-06 freeze on register_targets, no-leak after finalize, no-leak on WR-01 zero-expected path, cold-start empty dict, snapshot independence across targets registered at different times.
- `TestWeightedVoteBranch` (10 tests): MAJORITY default unchanged (2-of-3, 1-of-3); WEIGHTED high-weight dominates low-weight; **the STRT-01 SC1 differential test** — same `(True, False, False)` input produces `blocked=False` under MAJORITY and `blocked=True` under WEIGHTED with skewed weights; strict-greater tie-break at exact 50/50; cold-start 0.5 default for VPs missing from snapshot; zero-weight fallback to majority; warn-once-per-key (per-target, NOT per-country); WARNING log emission; vp_count preservation through the weighted branch.

`tests/infrastructure/test_orchestrator.py` — `_OrchestratorTestHelper.make_orchestrator()` initializes `_vp_outcomes = defaultdict(lambda: [0, 0])` so future orchestrator tests touching the WEIGHTED_VOTE path have the ledger pre-populated.

## STRT-01 SC1 Demonstration

The plan's must-have differential is captured by `TestWeightedVoteBranch.test_weighted_vs_majority_differential`:

| Input (vp_a, vp_b, vp_c blocked) | --aggregation majority | --aggregation weighted (weights 0.95, 0.05, 0.05) |
| -------------------------------- | ---------------------- | ------------------------------------------------- |
| (True, False, False)             | `blocked=False`        | `blocked=True`                                    |

Same input, same VPs, different `MeasurementResponse.blocked` value. This is the no-code-change behavior switch demanded by STRT-01 SC1.

## D-04 Bit-for-Bit Defaults

Omitting `--aggregation` (or passing `--aggregation majority`) resolves to `AggregationMethod.MAJORITY_VOTE`. The MAJORITY_VOTE branch computes `sum(b for _, b in votes_pairs) > len(votes_pairs) / 2`, which is numerically identical to the original `sum(votes) > len(votes) / 2`. `_vp_outcomes`, `_vp_weights`, and `_target_weights` are still mutated (defensive; they are cheap dicts), but never read because the WEIGHTED branch is not taken. Phase 02.1/03 baseline behavior preserved.

## Verification Performed

1. **Beta-smoothed helper checks** — `(0,0)→0.5`, `(1,0)→2/3`, `(0,1)→1/3`, `(99,1)→100/102`, `(5,5)→6/12=0.5`. PASS.
2. **`set_vp_weights` storage + defensive copy** — external mutation of input dict does not leak to internal state. PASS.
3. **Per-target snapshot at register_targets time** — a target registered before a weight push sees the OLD weights; subsequent push does not retroactively alter the snapshot. PASS.
4. **Cleanup pops** — `_target_weights` popped on both finalize paths (normal + WR-01 zero-expected). PASS.
5. **MAJORITY default 2-of-3 → True, 1-of-3 → False** (D-04 bit-for-bit). PASS.
6. **WEIGHTED high-weight dominates** — single 0.95-weight blocked vote outweighs two 0.05-weight not-blocked. PASS.
7. **WEIGHTED vs MAJORITY differential** — same input produces different `blocked` (STRT-01 SC1). PASS.
8. **Strict-greater tie-break** — exact 50/50 weighted sum gives `blocked=False` (matches majority semantics). PASS.
9. **Cold-start 0.5 default** — VPs missing from the per-target snapshot use 0.5 weight at finalize time. PASS.
10. **Zero-weight fallback** — all-zero weights produce a single WARNING log line per `(country, target)` and use majority for that finalization. PASS.
11. **Warn-once-per-key** — repeated zero-weight finalizations on the same key do not re-warn; different keys do warn. PASS.
12. **vp_count preservation** — `MeasurementResponse.vp_count == len(votes)` through the WEIGHTED branch. PASS.
13. **Full test suite** — `tests/infrastructure/` 173 passed, 1 deselected (pre-existing failure logged to `deferred-items.md`, unrelated to this work). PASS.

Default-run smoke against the orchestrator's `debug=True` entrypoint is deferred (requires `local/ev-certs.csv` not present in this worktree); behavior is exercised via the orchestrator-mock tests in `test_orchestrator.py` (preserved, no regressions).

## Deviations from Plan

None — plan executed exactly as written. Both tasks completed on first attempt with no Rule 1/2/3 auto-fixes required. The plan was very precise; the most subtle constraint (per-target snapshot timing relative to `register_targets`) was already called out explicitly in the action steps.

## Deferred Issues

One pre-existing test failure logged to `.planning/phases/04-strategy-variants/deferred-items.md`:

- `tests/infrastructure/test_orchestrator.py::TestVPRemovalOnEvalFailure::test_no_remove_when_all_pass` — fails on both this branch AND the wave-base commit (693b825). Unrelated to weighted-vote work; the test expects `remove_vantage_points` NOT to be called when an eval payload has a non-None template, but `_process_eval_results` does call it. Should be triaged separately.

## Commits

- `77cab18` — feat(04-02): add _vp_outcomes ledger + beta-smoothed VP weight push (Task 1)
- `bc70520` — feat(04-02): branch _try_finalize on aggregation_method for WEIGHTED_VOTE (Task 2)

## Downstream Wave Hooks

Wave 2 is independent of Waves 3/4/5. Future plans that touch this code:

- **Wave 5 (IN_ORDER PropagationMethod):** If IN_ORDER's held-batch flush propagates rewards in schedule order, the `_vp_outcomes` ledger will continue to receive controls-failed updates from `_check_vp_health` as raw results arrive — no interaction with hold semantics because outcomes are recorded per-raw-result, not per-finalized-MeasurementResponse.
- **Phase 5 (future, ASN strategy):** The Beta smoothing helper is reusable for any per-entity reliability signal (VP IP, ASN, country) — `_beta_smoothed_weight` is a `@staticmethod` for this reason.
- **v2 ADVS-02 (now partially shipped):** Phase 4 Plan 02 covers the per-VP weight half. The other half (cross-iteration weight decay, ASN-aware grouping) remains v2 scope per CONTEXT.md decision boundary.

## Threat Flags

None — this plan introduces no new network endpoints, no new file writes, no new schema changes at trust boundaries. The only new shared state (`_target_weights`, `_weighted_fallback_warned`) is protected by the aggregator's existing `self._lock`. T-04-06..T-04-10 from the plan's threat model are all addressed:

- T-04-06 (Tampering): existing pipeline trust assumption unchanged; Beta bounds weights to (0, 1), single VP cannot dominate without long skew.
- T-04-07 (Information Disclosure): `_vp_outcomes` holds operational IPs already in INFO logs; no PII added.
- T-04-08 (DoS, numerical): zero-weight fallback covers the only division-by-zero path; strict `total_weight == 0.0` check.
- T-04-09 (Race): both `set_vp_weights` and the new `_try_finalize` paths take `self._lock`; `register_targets` snapshots `_target_weights[key]` under the same lock; WR-06 invariant preserved.
- T-04-10 (Repudiation): weights are derivable from `_vp_outcomes`; partial state is best-effort per Phase 02 D-12.

## Self-Check: PASSED

- `Infrastructure/main/orchestrator.py` — FOUND (modified, _vp_outcomes + _beta_smoothed_weight + _compute_vp_weights + set_vp_weights call present)
- `Infrastructure/utils/aggregator.py` — FOUND (modified, set_vp_weights + _target_weights + WEIGHTED_VOTE branch + _weighted_fallback_warned present)
- `tests/infrastructure/test_aggregator.py` — FOUND (modified, 19 new tests, 32 total passing)
- `tests/infrastructure/test_orchestrator.py` — FOUND (modified, _vp_outcomes added to make_orchestrator helper)
- Commit `77cab18` — FOUND in `git log --oneline 693b825..HEAD`
- Commit `bc70520` — FOUND in `git log --oneline 693b825..HEAD`
