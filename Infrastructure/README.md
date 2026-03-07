# Infrastructure

This folder contains the core infrastructure for a distributed measurement and learning system. It is organized around **nodes**, **orchestrator**, **API**, and **learning models**, with supporting utilities.

The design emphasizes modularity, extensibility, and suitability for research-driven experimentation in distributed systems and adaptive measurement.

---

## How to Run

1. Start the Hyperquack worker-server program (must be modified to make post requests)
2. Ensure the Infrastructure folder is within the CenRL folder, and that the CenRL module has been installed
3. Run from the orchestrator

Example: `python3 Infrastructure/main/orchestrator.py -E 1 -m 1000 -v -f "categories" -a inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv -f "categories" -s 0.0 -c 0.03 -V 0.0 -o outputs/outtest`

---

## Module Structure

```
Infrastructure/
├── apis/
│   ├── api.py
│   ├── funneler.py
│   ├── server.py
│   └── README.md
│
├── examples/
│   ├── vps.py
│   └── README.md
│
├── main/
│   ├── node.py
│   ├── orchestrator.py
│   ├── vantage_points.py
│   ├── test_vantage_points.py
│   └── README.md
│
├── models/
│   ├── BatchUCB.py
│   └── README.md
│
├── utils/
│   ├── aggregator.py
│   ├── eval_store.py
│   ├── store.py
│   ├── structures.py
│   ├── test_aggregator.py
│   └── README.md
│
└── README.md
```

---

## High-Level Execution Flow

1. The **Orchestrator** creates a `VantagePoints` pool from `ev-certs.csv` and a CIDR blocklist, then draws initial VPs for each target country.
2. Drawn VPs are sent to the **Go server** (Hyperquack) for evaluation via the **Funneler** API layer.
3. The **Server** receives evaluation results on `/vp-evaluated` and buffers them in the **EvalStore**.
4. The Orchestrator processes eval results each tick:
   - Good VPs are confirmed active; a **RegionalNode** is created lazily on the first good VP for a country.
   - Bad VPs are rejected and replaced from the inactive pool.
5. For each active node, the Orchestrator:
   - Drains raw measurement results from the **MeasurementStore**.
   - Checks VP health (removes VPs that exceed a failure threshold).
   - Feeds per-VP results into the **MeasurementAggregator**, which waits for all active VPs to report per target.
   - Delivers majority-vote aggregated results to the node's **BatchUCB** model.
   - Requests new target URLs from the model and schedules measurements across all active VPs.
6. The loop repeats until all countries have completed their episodes.

---

## Notes

- This module focuses on infrastructure rather than deployment details.
- Logging, persistence backends, authentication, and security layers are expected to be configured externally or extended as needed.
- The structure is designed to support experimentation, iteration, and scaling.
- Debug mode (`debug=True` on the Orchestrator) simulates the full pipeline without a Go server, including synthetic VP evaluations and measurements.

---
