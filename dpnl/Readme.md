# Dynamic Probabilistic NeuroSymbolic Logic (DPNL)

This project provides a Python implementation of **Dynamic Probabilistic NeuroSymbolic Logic (DPNL)** — a flexible framework to perform symbolic reasoning over **probabilistic inputs**. It supports custom symbolic functions, logic-based reasoning, and includes comparisons to external systems like **ProbLog**.

---

## Project Structure

| File | Description |
|------|-------------|
| `dpnl.py` | Core implementation of DPNL: probabilistic variables, recursive inference, oracles, and logic integration. |
| `graph_reachability.py` | Contains the `graph_reachability` symbolic function and a custom optimized oracle for reasoning over uncertain graphs. |
| `z3_logic.py` | Encodes symbolic functions in first-order logic using Z3 and builds corresponding DPNL-compatible oracles. |
| `dpnl_vs_problog.py` | Experimental script that compares DPNL (in multiple modes) against ProbLog for graph reachability inference. |

---

## What is DPNL?

DPNL is a form of **neurosymbolic inference** that:
- Handles structured inputs (`X`) with discrete random variables (`RndVar`)
- Applies a symbolic function `S(X)`
- Computes `P(S(X) == output)` using a dynamic recursive algorithm
- Uses **oracles** to shortcut reasoning when full inference isn't needed

It supports:
- **Automatic oracle construction** (e.g., from a Python function `S`)
- **Logic-based symbolic functions** (`LogicS`) with provers (e.g., Z3)
- **Custom heuristics** for branching decisions

---

# Running the Correctness Experiments

The correctness experiment is in `dpnl_vs_problog.py`.
It compares the output probability results (and time of computation) of :

- `DPNL (Basic Oracle)`: automatic oracle based on early evaluation
- `DPNL (Optimized Oracle)`: handcrafted oracle with domain-specific optimizations
- `DPNL (Logic Oracle)`: symbolic function encoded in logic (e.g., Z3)
- `ProbLog`: a logic programming system with probabilistic inference

### To Run:
```bash
python dpnl_vs_problog.py
```

This runs multiple trials of random graphs and prints:

```
N   | DPNL Basic Oracle | DPNL Hand-Crafted Oracle | DPNL Logic Oracle | ProbLog
----|-------------------|--------------------------|-------------------|--------
3   | 0.7890 (0.01s)     | 0.7890 (0.01s)           | 0.7890 (0.02s)    | 0.7889 (0.90s)
...
```

⚠️ Note: This may take **10-20 minutes** to complete.

---

## Features

- Generic symbolic reasoning over probabilistic inputs
- Multiple oracle types for inference speed trade-offs
- Logic-based reasoning using Z3 SMT solver
- Oracle-guided branching with heuristics
- Correctness comparison against ProbLog

---

## Requirements

- Python 3.8+
- [z3-solver](https://pypi.org/project/z3-solver/)
- [ProbLog](https://dtai.cs.kuleuven.be/problog/) installed and available via CLI (optional, for comparison)

Install dependencies:
```bash
pip install z3-solver
```