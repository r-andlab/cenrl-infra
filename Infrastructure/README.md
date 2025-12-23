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
│   └── __init__.py
│
├── examples/
│   ├── vps.py
│   └── __init__.py
│
├── main/
│   ├── node.py
│   ├── orchestrator.py
│   └── __init__.py
│
├── models/
│   ├── BatchUCB.py
│   └── __init__.py
│
├── utils/
│   ├── store.py
│   ├── structures.py
│   └── __init__.py
│
└── README.md
```


---

## Module Breakdown

### `apis/`

Defines the API layer used for communication between node agents and Hyperquack.

- **`api.py`**  
  Core API definitions and request handling logic. (No longer important)

- **`server.py`**  
  Server entry point responsible for accepting incoming measurement results from Hyperquack. Stores results within a threadlocked queue.

- **`funneler.py`**  
  Controls all communication with Hyperquack. Parses and returns measurement results from threadlocked queue when requested.

This layer cleanly separates networking and protocol concerns from system logic.

---

### `main/`

Contains the primary runtime logic of the system.

- **`node.py`**  
  Each node controls an instance of the model (agent). Nodes store member variables for each agent, and can request and absorb measurements.

- **`orchestrator.py`**  
  Coordinates nodes, schedules work, and interfaces with learning models.

This directory governs how the system executes and scales.

---

### `models/`

Learning and decision-making components.

- **`BatchUCB.py`**  
  Implements a batch Upper Confidence Bound (UCB) model for adaptive selection and exploration across repeated measurements or actions. Inherits from UCBNaive.

Models are intentionally isolated from networking and storage layers to reduce complexity.

---

### `utils/`

Shared utilities and core abstractions.

- **`structures.py`**  
  Common data structures, payload schemas, and shared types used throughout the system.

- **`store.py`**  
  Storage abstraction for managing results.

These utilities are dependency-light and reused across multiple modules.

---

### `examples/`

Reference scripts and minimal examples.

- **`vps.py`**  
  A dictionary of example vantage points for the orchestrator to use..

This directory is intended for experimentation and onboarding rather than production use.

---


## High-Level Execution Flow

1. Create API instance and run listener in the background
2. Launch one or more worker nodes
3. Run the orchestrator to assign tasks (loops through all nodes forever or until all have reported finishing)
   1. If the node can request measurements, it asks the orchestrator
   2. If the node is awaiting measurements, orchestrator checks if there are any results to report
4. Nodes perform work and report results
5. Models update decisions based on collected data
6. Repeat

---

## Notes

- This module focuses on infrastructure rather than deployment details.
- Logging, persistence backends, authentication, and security layers are expected to be configured externally or extended as needed.
- The structure is designed to support experimentation, iteration, and scaling.
- It currently requires code modification to edit batch size, and I need to figure out exactly how to store results in the best way possible.

---

