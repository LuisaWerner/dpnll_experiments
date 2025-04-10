import random
from dpnl import (
    BoolRndVar, PNLProblem, RndVar,
    BasicOracle, unknown, Oracle
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
# Complete and Optimized Oracle for Graph Reachability
# =====================================

class OptimizedGraphReachabilityOracle(Oracle):
    def __init__(self):
        super().__init__(graph_reachability)
        self.path_when_unknown_is_true = None

    def _find_path(self, g: list[list[RndVar]], src:int, dst:int, consider_unknown_as:bool):

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
                    # We slightly modify the classic alogrithm to take into consider wheter we consider unknown as
                    # true or false
                    if g[cur][node].defined():
                        if g[cur][node].value:
                            path.append(node)
                            dfs()
                    elif consider_unknown_as:
                        path.append(node)
                        dfs()
            path.pop()

        try:
            dfs()
            return None
        except FoundPath:
            return path

    def __call__(self, X: tuple[list[list[RndVar]], int, int], S_output: bool):
        self.path_when_unknown_is_true = None
        g, src, dst = X
        path_when_unknown_is_true = self._find_path(g, src, dst, consider_unknown_as=True)
        path_when_unknown_is_false = self._find_path(g, src, dst, consider_unknown_as=False)

        res = unknown
        if path_when_unknown_is_true is None:
            # Whatever the valuations of the non valuated variables there is no path then we return False
            res = False
        if path_when_unknown_is_false is not None:
            # Whatever the valuations of the non valuated variables there is a path then we return True
            res = True
        # If there are only path from src to dst that contains non-valuated edges

        if res == unknown:
            # Saving information for the choosing heuristic
            self.path_when_unknown_is_true = path_when_unknown_is_true

        # We switch the result according to the expected output from graph reachability
        if res != unknown and S_output is False:
            res = not res

        return res

    def choose_variable_heuristic(self, X, S_output):
        g, src, dst = X
        path = self.path_when_unknown_is_true
        for i in range(len(path) - 1):
            edge_var = g[path[i]][path[i + 1]]
            if not edge_var.defined():
                return edge_var
        print("DEBUG PATH:", path)
        print("DEBUG EDGE STATES:", [[var.value for var in row] for row in g])
        raise AssertionError("No undefined variable found on path — inconsistency in oracle logic")


# =====================================
# Tests and Demonstration
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
    return [[BoolRndVar(f'{i}-{j}', random.uniform(0.0, 1.0), value=random.choice(choice_list))
             for j in range(N)] for i in range(N)]


def print_graph_matrix(g):
    print("------ Graph edges ------")

    for i in range(len(g)):
        for j in range(len(g)):
            if g[i][j].value is True:
                print(f"{i}--->{j}")
            elif not g[i][j].defined():
                print(f"{i}-{g[i][j].domain_distrib[True]:.2f}->{j}")

    print("-------------------------")


def compare_oracles(g, src, dst):
    basic = BasicOracle(graph_reachability)
    complete = OptimizedGraphReachabilityOracle()

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

    prob_basic = pb.prob(
        BasicOracle(graph_reachability),
        True
    )

    prob_optimised = pb.prob(
        OptimizedGraphReachabilityOracle(),
        True
    )

    print(f"Each unknown edge of the graph has random probability.")
    print(f"Estimated probability path exists from 0 to 3 with basic oracle: {prob_basic:.4f}")
    print(f"Estimated probability path exists from 0 to 3 with optimized oracle: {prob_optimised:.4f}")
