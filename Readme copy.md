
# DPNL

DPNL is a Python framework that combines logic-based symbolic reasoning with probabilistic modeling.
It provides an extensible architecture for defining, executing, and optimizing inference over symbolic functions that depend on uncertain inputs.

This repository includes the DPNL core engine, logic integrations, inference oracles, caching strategies, and real-world examples like graph reachability and MNIST-style symbolic addition.

---

## Features

- Define symbolic functions with probabilistic inputs
- Choose from multiple oracles for inference control (enumerative, logic-based, hand-crafted)
- Perform exact or approximate probabilistic reasoning
- Support for SAT logic and integration with external tools like ProbLog
- Built-in caching to accelerate repeated inference
- Example problems to demonstrate effectiveness

---

## Installation & Dependencies

To use or extend this project, ensure you have the following Python packages installed:

```bash
pip install numpy networkx pysat cloudpickle
```

> Additionally, to run ProbLog-related comparisons, install ProbLog 2:
- [https://dtai.cs.kuleuven.be/problog/](https://dtai.cs.kuleuven.be/problog/)

---

## Project Structure

```
DPNL/
├── dpnl/                  # Core implementation of DPNL
├── GRAPH-REACHABILITY/    # DPNL applied to probabilistic graph path finding
├── MNIST-N-SUM/           # DPNL applied to probabilistic symbolic addition
```

---

## Example Use Cases

### Graph Reachability

Estimate the probability that a path exists from source to destination in a graph where edges may or may not exist.

```bash
cd dpnl/GRAPH-REACHABILITY
python run.py
```

### MNIST-N-SUM

Estimate the probability that two digit sequences (e.g., from MNIST) sum to a specific value.

```bash
cd dpnl/MNIST-N-SUM
python run.py
```

---

## 🛠 Extendability

You can:
- Implement new symbolic functions (`dpnl/core/symbolic.py`)
- Create your own oracle (`dpnl/oracles/`)
- Add logic encodings (`dpnl/logics/`)
- Define caching strategies (`dpnl/cachings/`)

---

## License

This project is open for academic and research use. For commercial licensing or inquiries, please contact the author(s).

