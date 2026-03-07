# models/

Learning and decision-making components. Isolated from networking and storage layers.

---

## Modules

### `BatchUCB.py` — `BatchUCB`

Extends `UCBNaive` (from `models.ucb.ucb_naive`) to implement batch-mode target selection and measurement absorption for the multi-armed bandit framework.

**Key methods:**

| Method | Purpose |
|---|---|
| `queue_measurement(batch_size)` | Select an arm via UCB policy, then choose up to `batch_size` targets from it. Returns a list of target URLs and tracks them as `Pending` objects. |
| `absorb_measurement(measurement)` | Process a `MeasurementResponse` (as dict). Matches it to a pending measurement, computes reward, and propagates to the model. |
| `choose_targets(arm, batch_size)` | Implements `TOP_K_FROM_ARM` selection: picks the top-k targets from the chosen arm's action space node. |

**Selection methods (configured via `BatchSelectionMethod`):**

- `TOP_K_FROM_ARM` — Select the top-k targets from a single arm (default)
- `UNIFORM_SPREAD` / `WEIGHTED_SPREAD` — Alternative strategies (available but less commonly used)

**Batch size methods (configured via `BatchSizeMethod`):**

- `CONSTANT_VAL` — Fixed batch size each step
- `VARY_ON_SUCCESS` — Adjust batch size based on measurement outcomes
- `NESTED_RL` — Use a nested RL agent to learn the optimal batch size

**Propagation methods (configured via `PropagationMethod`):**

- `ON_RECEIPT` — Propagate rewards as soon as each measurement is received
- `IN_ORDER` — Buffer and propagate rewards in the order they were requested

The model maintains an action space tree where each node represents a category/feature combination, and leaves represent individual measurement targets (URLs). The UCB algorithm balances exploration of under-tested categories with exploitation of categories showing high blocking rates.

---
