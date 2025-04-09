import random
from dpnl import (
    BoolRndVar, PNLProblem, RndVar,
    basic_oracle, basic_oracle_choose_heuristic,
    unknown
)


# =====================================
# 📌 Symbolic Function: Graph Reachability
# =====================================

def graph_reachability(X: tuple[list[list[RndVar]], int, int]):
    """
    Symbolic function that determines whether a path exists from node `src` to node `dst`
    in a graph represented by a matrix of RndVar (boolean-valued edge existence).
    """
    g, src, dst = X
    n = len(g)
    seen = set()
    working = [src]
    while working:
        cur = working.pop()
        if cur == dst:
            return True
        if cur not in seen:
            seen.add(cur)
            for node in range(n):
                if g[cur][node].value:
                    working.append(node)
    return False


# =====================================
# 🔮 Complete Oracle for Graph Reachability
# =====================================

def graph_reachability_complete_oracle(X: tuple[list[list[RndVar]], int, int], S_output: bool):
    """
    A complete oracle that forces all undefined edge variables to both True and False
    to check whether the result is determined under all completions.
    """
    g, src, dst = X

    # Replace unknowns with True
    unknown2true_g = [
        [BoolRndVar(var.name, var.domain_distrib[True], value=var.value if var.defined() else True)
         for var in row] for row in g
    ]

    # Replace unknowns with False
    unknown2false_g = [
        [BoolRndVar(var.name, var.domain_distrib[True], value=var.value if var.defined() else False)
         for var in row] for row in g
    ]

    result_if_true = graph_reachability((unknown2true_g, src, dst))
    result_if_false = graph_reachability((unknown2false_g, src, dst))

    if result_if_false is True:
        return True if S_output else False
    if result_if_true is False:
        return False if S_output else True

    return unknown


# =====================================
# 🎯 Heuristic for Selecting Most Informative Edge
# =====================================

def graph_reachability_complete_oracle_choose_heuristic(pnl_problem: PNLProblem, S_output, oracle_answer):
    """
    Chooses an undefined edge variable that lies on a path from `src` to `dst`
    composed only of edges that are True or Unknown.
    """
    g, src, dst = pnl_problem.X
    seen = set()
    path = [src]

    class FoundPath(Exception):
        pass

    def dfs():
        cur = path[-1]
        if cur not in seen:
            seen.add(cur)
            if cur == dst:
                raise FoundPath()
            for node in range(len(g)):
                if g[cur][node].value is not False:
                    path.append(node)
                    dfs()
        path.pop()

    try:
        dfs()
        raise AssertionError("No path found, but oracle indicated uncertainty")
    except FoundPath:
        for i in range(len(path) - 1):
            edge_var = g[path[i]][path[i + 1]]
            if not edge_var.defined():
                return edge_var

        print("DEBUG PATH:", path)
        print("DEBUG EDGE STATES:", [[var.value for var in row] for row in g])
        raise AssertionError("No undefined variable found on path — inconsistency in oracle logic")


# =====================================
# 🧪 Tests and Demonstration
# =====================================

def graph_map_to_RndVar(g):
    N = len(g)
    return [[BoolRndVar(f'{i}-{j}', 0.5, value=g[i][j]) for j in range(N)] for i in range(N)]


def random_graph(N: int, proportion_of_missing_edge: float = 0.0):
    """
    Generate a random NxN graph, where each edge is:
    - `unknown` with probability determined by `proportion_of_missing_edge`
    - `False` otherwise
    """
    choice_list = [unknown]
    if proportion_of_missing_edge < 1.0:
        choice_list = [False] * max(0, int(1 / (1 - proportion_of_missing_edge)) - 1) + [unknown]
    return [[BoolRndVar(f'{i}-{j}', 0.5, value=random.choice(choice_list))
             for j in range(N)] for i in range(N)]


def print_graph_matrix(g):
    print("------ Graph edges ------")

    for i in range(len(g)):
        for j in range(len(g)):
            if g[i][j].value is True:
                print(f"{i}--->{j}")
            elif not g[i][j].defined():
                print(f"{i}-?->{j}")

    print("-------------------------")


def compare_oracles(g, src, dst):
    basic = basic_oracle(graph_reachability)
    complete = graph_reachability_complete_oracle

    print_graph_matrix(g)
    print(f"Basic Oracle  (0 → {dst}):", basic((g, src, dst), True))
    print(f"Complete Oracle (0 → {dst}):", complete((g, src, dst), True))
    print()


if __name__ == '__main__':
    print("\n=== Test 1: No Path Expected ===")
    g = [
        [True, False, False],
        [True, unknown, unknown],
        [True, unknown, unknown],
    ]
    g = graph_map_to_RndVar(g)
    compare_oracles(g, 0, 1)
    compare_oracles(g, 0, 2)

    print("\n=== Test 2: Definite Path Exists ===")
    g = [
        [False, True, False],
        [True, False, True],
        [unknown, unknown, False],
    ]
    g = graph_map_to_RndVar(g)
    compare_oracles(g, 0, 1)
    compare_oracles(g, 0, 2)

    print("\n=== Test 3: Oracle Undecided For The Basic Oracle (due to unknown edges) ===")
    g = [
        [True, True, unknown],
        [False, unknown, True],
        [unknown, unknown, unknown],
    ]
    g = graph_map_to_RndVar(g)
    compare_oracles(g, 0, 1)
    compare_oracles(g, 0, 2)

    print("\n=== Test 4: Full DPNL Inference (with branching) ===")

    g = random_graph(6, 0.7)
    pb = PNLProblem((g, 0, 3), graph_reachability)

    print("Graph for DPNL Inference (T = True, F = False, ? = Unknown):")
    print_graph_matrix(g)

    prob = pb.prob(
        basic_oracle(pb.S),
        True,
        choose=basic_oracle_choose_heuristic
    )

    print(f"Each unknown edge is considered to have a probability of being present of 0.5")
    print(f"Estimated probability path exists from 0 to 3: {prob:.4f}")
