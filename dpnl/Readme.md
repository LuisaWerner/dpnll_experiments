# DPNL

**DPNL** is a Python framework for combining logic-based symbolic reasoning with probabilistic inputs.
It allows users to model complex problems involving uncertainty, logic, and structure — such as graph analysis, symbolic addition, and more — using a declarative yet efficient framework.

---

## 📁 Folder Structure

| Folder                | Description |
|-----------------------|-------------|
| `core/`               | Core abstractions of the DPNL framework, including symbolic inputs, probabilistic variables (`RndVar`), oracle definitions, the main `PNLProblem` inference engine, and caching interfaces. |
| `oracles/`            | Implements different oracle strategies, such as basic evaluation-based oracles, enumeration-based oracles, and formal logic-driven oracles used to guide branching in probabilistic inference. |
| `logics/`             | Provides the bridge between DPNL and external logic systems. Currently supports SAT logic, allowing symbolic problems to be encoded as SAT instances for use with logic-based oracles. |
| `cachings/`           | Defines hashing strategies to cache intermediate results in DPNL. Contains SAT-based component-caching using unit propagation to avoid redundant computation. |


---

## 🔧 Core Components

### `core/`
Defines the foundational abstractions:
- `Input` and `Symbolic` for probabilistic symbolic functions
- `RndVar` for random variables with unknown values
- `Oracle` and `CachingHash` for decision-making and optimization
- `PNLProblem` for executing DPNL inference

---

### `logics/`
Interfaces with symbolic logic systems:
- Currently supports Boolean logic using SAT solvers via the `pysat` library
- Provides symbolic-to-SAT conversion and clause generation

---

### `oracles/`
Implements different oracle strategies:
- `BasicOracle` executes symbolic functions directly and observes partial evaluations
- `EnumerationOracle`, `CombinationOracle`, and `LogicOracle` offer flexibility and formal logic-based reasoning

---

### `cachings/`
Efficient caching of DPNL evaluations:
- `SATCachingHash` reduces redundant computation by hashing simplified SAT clause representations using unit propagation