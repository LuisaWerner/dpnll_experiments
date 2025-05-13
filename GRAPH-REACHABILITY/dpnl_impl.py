from typing import Any
import networkx as nx

from dpnl.core.oracle import Oracle
from dpnl.core.symbolic import Input, Symbolic
from dpnl.core.variable import RndVar, unknown, BoolRndVar
from dpnl.core.problem import PNLProblem
from dpnl.oracles.enumeration import EnumerationOracle
from dpnl.oracles.basic import BasicOracle
from dpnl.oracles.logic import LogicOracle
from dpnl.logics.sat_logic import SATLogic, SATLogicSymbolic
from dpnl.cachings.sat_logic_hash import SATCachingHash


class GraphReachInput(Input):
    def __init__(self, size: int, src: int, dst: int):
        assert 0 <= src < size and 0 <= dst < size
        self.graph = [
            [BoolRndVar(("graph", i, j), 0.5) for j in range(size)] for i in range(size)
        ]
        self.src = src
        self.dst = dst
        self.size = size
        super().__init__(probabilistic_attributes={"graph"})


class GraphReachSymbolic(Symbolic):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __call__(self, I: GraphReachInput):
        seen = set()
        working = [I.src]
        while working:
            cur = working.pop()
            if cur == I.dst:
                return True
            if cur not in seen:
                seen.add(cur)
                for node in range(I.size):
                    if I.graph[cur][node].value:
                        working.append(node)
        return False


class GraphReachLogicSymbolic(SATLogicSymbolic):

    def __init__(self, size: int):
        super().__init__([])
        assert isinstance(self.logic, SATLogic)  # For pycharm typechecking

        # Setting internal variables
        self.size = size

        # Input
        self.graph = [
            [self.logic.lit(("graph", i, j)) for j in range(size)] for i in range(size)
        ]
        self.src = self.var("src", range(size))
        self.dst = self.var("dst", range(size))

        # Output
        self.success = self.success = self.logic.lit("success")

        # Intermediate values

        # The set of node reached from src : reached[dst] => success
        self.reached = [self.logic.lit(("reached", node)) for node in range(size)]

        # A possible path from src : success => dst in the path
        self.path = [self.var(f"path[{idx}]", range(-1, size)) for idx in range(size)]

        # Computing reached
        for node1 in range(size):
            self.imply([self.src == node1], [self.reached[node1]])
            for node2 in range(size):
                self.imply(
                    [self.reached[node1], self.graph[node1][node2]],
                    [self.reached[node2]]
                )

        # Forcing path to be a valid path from src
        self.path[0].equal(lambda x: x, self.src)
        for idx in range(size - 1):
            self.imply([self.path[idx] == -1], [self.path[idx + 1] == -1])
            for node1 in range(size):
                for node2 in range(size):
                    self.imply(
                        [self.path[idx] == node1, self.path[idx + 1] == node2],
                        [self.graph[node1][node2]]
                    )

        for node in range(size):
            # reached[dst] => success
            self.imply(
                [self.dst == node, self.reached[node]],
                [self.success]
            )
            # success => dst in path
            self.imply(
                [self.success, self.dst == node],
                [self.path[idx] == node for idx in range(size)]
            )

    def assumptions_linked_to(self, var: RndVar) -> list[Any]:
        assert isinstance(var.name, tuple) and len(var.name) == 3 and var.name[0] == "graph"
        edge_lit = self.graph[var.name[1]][var.name[2]]
        return [edge_lit]

    def assumptions(self, I: GraphReachInput) -> list[Any]:
        assumptions = [self.src == I.src, self.dst == I.dst]
        for i in range(self.size):
            for j in range(self.size):
                var = I.graph[i][j]
                if var.value is True:
                    assumptions.append(self.graph[i][j])
                elif var.value is False:
                    assumptions.append(-self.graph[i][j])
        return assumptions

    def conclusion(self, I: GraphReachInput) -> Any:
        return self.success


# PNL problem

def problem(I: GraphReachInput):
    return PNLProblem(I, GraphReachSymbolic(I.size))


# Oracles

def enumeration(length: int):
    return EnumerationOracle(GraphReachSymbolic(length))


def basic(length: int):
    return BasicOracle(GraphReachSymbolic(length))


def logic(length: int):
    return LogicOracle(GraphReachLogicSymbolic(length))


def hand_crafted(length: int):
    return HandCraftedOracle(length)


class HandCraftedOracle(Oracle):
    def __init__(self, size: int):
        super().__init__(GraphReachSymbolic(size))
        self.path_when_unknown_is_true = None

    def _find_path(self, I: GraphReachInput, consider_unknown_as: bool):

        seen = set()
        path = [I.src]

        class FoundPath(Exception):
            pass

        def dfs():
            cur = path[-1]
            if cur not in seen:
                seen.add(cur)
                if cur == I.dst:
                    raise FoundPath()
                for node in range(I.size):
                    # We slightly modify the classic alogrithm to take into consider wheter we consider unknown as
                    # true or false
                    if I.graph[cur][node].defined():
                        if I.graph[cur][node].value:
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

    def __call__(self, I: GraphReachInput, S_output: bool):
        self.path_when_unknown_is_true = None
        path_when_unknown_is_true = self._find_path(I, consider_unknown_as=True)
        path_when_unknown_is_false = self._find_path(I, consider_unknown_as=False)

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

    def choose_variable_heuristic(self, I: GraphReachInput, S_output):
        path = self.path_when_unknown_is_true
        for i in range(len(path) - 1):
            edge_var = I.graph[path[i]][path[i + 1]]
            if not edge_var.defined():
                return edge_var


# Caching Hash

def sat_logic_hash(length: int):
    return SATCachingHash(GraphReachLogicSymbolic(length))
