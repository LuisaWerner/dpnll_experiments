import random

from dpnl.core.oracle import Oracle
from dpnl.core.variable import unknown


class CombinationOracle(Oracle):
    def __init__(self, oracle_list: list[Oracle]):
        super().__init__(oracle_list[0].S)
        self.oracle_list = oracle_list

    def __call__(self, X, S_output):
        result = unknown
        for oracle in self.oracle_list:
            result = oracle(X, S_output)
            if result is not unknown:
                break
        return result

    def choose_variable_heuristic(self, X, S_output):
        """
        A simple variable selection heuristic:
        - Directly uses the variable that caused the oracle to return Unknown
        - Assumes this is the most informative branching point
        """
        return self.oracle_list[len(self.oracle_list)-1].choose_variable_heuristic(X, S_output)