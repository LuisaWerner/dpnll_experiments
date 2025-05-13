# Overview of DPNL 
This repository contains the code base for the paper `DPNL: A DPLL-based Algorithm for Probabilistic Neurosymbolic Learning`.
It the implementation of DPNL, the result files and the code to reproduce the baselines reported for DPNL. 
It has the following file structure. 

* The folder `dpnl` contains a Python implementation of **Dynamic Probabilistic NeuroSymbolic Logic (DPNL)** — a flexible framework to perform symbolic reasoning over **probabilistic inputs**. It supports custom symbolic functions, logic-based reasoning, and includes comparisons to external systems like **ProbLog**. Instructions to run the code are given in an additional `dpnl/Readme.md` file.
* The directories `GRAPH_REACHABILTY` and ` MNIST_N_SUM` contain the source code of the symbolic experiments
* The directory `MNIST_N_SUM (PNL)` contains the source code of the neurosymbolic experiments
* We provide the cloned repositories that we used as baselines and provide the configurations with which we obtained the results reported in the paper. 
* We further provide the result summary files of DPNL and the baselines on the neurosymbolic task.
The citations of the baselines are listed below. 

```
├── baseline_repositories/
|    ├── a-nesi/
|    |   └── anesi/
|    |       └── experiments/
|    |            └── mnist_op/
|    |                ├──anesi_mnist_add.py
|    |                └── baseline_dpnl 
|    |                    └── test_add_no_prune_explain.yaml
|    |                    └── test_add_predict_only.yaml
|    |                    └── test_add_prune.yaml
|    ├── deepproblog/
|    │   └── src/
|    |        └── deepproblog/
|    |            └── examples/
|    |                └── MNIST/
|    |                   └── addition.py 
|    └──scallop/
|       └── experiments/
|            └── mnist/
|               └── sum_2_baseline.py
├──result_logs/
|    └──a-nesi
|    └──deepproblog
|    └──dpnl
|    └──scallop
|
├──dpnl
|	 └──core
|	 └──logics
|	 └──oracles
|	 └──Readme.md
|	 └──...
|
├──GRAPH-REACHABILTY
├──MNIST-N-SUM
├──MNIST-N-SUM (PNL)

	 
```

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
├── MNIST-N-SUM (PNL)/     # DPNL applied to neurosymbolic addition
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


# Reproduction of baselines 

### A-Nesi
The repository has been cloned from [https://github.com/HEmile/a-nesi](https://github.com/HEmile/a-nesi).
1. Follow the instructions in `baseline_repositories/a-nesi/README.md` to set up the environment.
2. The configs filles that we used to reproduce the results are contained in `baseline_repositories/a-nesi/anesi/experiments/mnist_op/baselines_dpnl`. As listed above, we run experiments for the variants for predict, explain, and prune. 
3. To run these experiments, replace the key `entity` by your weights and biases entity name 
4. run
```
cd baseline_repositories/a-nesi/anesi/experiments/mnist_op
wandb sweep baseline_dpnl/test_predict_only.yaml % replace name for other files
wandb agent <sweep_id>
```

### DeepProbLog and DPLA*
The code for DeepProblog and DPLA* is contained in the same repository.
It has been cloned from [https://github.com/ML-KULeuven/deepproblog](https://github.com/ML-KULeuven/deepproblog).
1. Follow the instructions in `baseline_repositories/deepproblog/README.md` to set up the environment.
2. The script to run is `.../src/deepproblog/examples/MNIST/addition.py`.
3. For DeepProblog run ```python addition.py 1 True``` (1 is the value for N, choose it in [1,4])
4. For DPLA* run ```python addition.py 1 False``` (1 is the value for N, choose it in [1,4])


### Scallop (exact) and Scallop (k=3)
The repository has been cloned from [https://github.com/scallop-lang/scallop](https://github.com/scallop-lang/scallop)
1. Follow the instructions in `.../scallop/readme.md` to set up the environment
2. The script to run is `.../scallop/experiments/mnist/sum_2_baseline.py`
3. For Scallop (exact) run `python sum_2_baseline.py 1 True` (1 is the value for N, choose it in [1,4])
4. For Scallop (k=3) run `python sum_2_baseline.py 1 False` 1 is the value for N, choose it in [1,4])

## Result Summaries
We further provide the results obtained by running the baselines and DPNL on the MNIST benchmark. 
The files can be found in subdirectories of `.../result_logs`.


## Baseline Citations

```
# A-Nesi 
@inproceedings{NEURIPS2023_4d9944ab,
 author = {van Krieken, Emile and Thanapalasingam, Thiviyan and Tomczak, Jakub and van Harmelen, Frank and Ten Teije, Annette},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {24586--24609},
 publisher = {Curran Associates, Inc.},
 title = {A-NeSI: A Scalable Approximate Method for Probabilistic Neurosymbolic Inference},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/4d9944ab3330fe6af8efb9260aa9f307-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}

# DeepProbLog
@inproceedings{NEURIPS2018_dc5d637e,
 author = {Manhaeve, Robin and Dumancic, Sebastijan and Kimmig, Angelika and Demeester, Thomas and De Raedt, Luc},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {DeepProbLog:  Neural Probabilistic Logic Programming},
 url = {https://proceedings.neurips.cc/paper_files/paper/2018/file/dc5d637ed5e62c36ecb73b654b05ba2a-Paper.pdf},
 volume = {31},
 year = {2018}
}

# DPLA*
@inproceedings{KR2021-45,
    title     = {{Approximate Inference for Neural Probabilistic Logic Programming}},
    author    = {Manhaeve, Robin and Marra, Giuseppe and De Raedt, Luc},
    booktitle = {{Proceedings of the 18th International Conference on Principles of Knowledge Representation and Reasoning}},
    pages     = {475--486},
    year      = {2021},
    month     = {11},
    doi       = {10.24963/kr.2021/45},
    url       = {https://doi.org/10.24963/kr.2021/45},
  }


# Scallop 
@inproceedings{NEURIPS2021_d367eef1,
 author = {Huang, Jiani and Li, Ziyang and Chen, Binghong and Samel, Karan and Naik, Mayur and Song, Le and Si, Xujie},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {25134--25145},
 publisher = {Curran Associates, Inc.},
 title = {Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/d367eef13f90793bd8121e2f675f0dc2-Paper.pdf},
 volume = {34},
 year = {2021}
}

```


# License

This project is open for academic and research use. For commercial licensing or inquiries, please contact the author(s).