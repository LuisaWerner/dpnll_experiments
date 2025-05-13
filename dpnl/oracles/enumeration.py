# Naive oracle
from dpnl.core.oracle import Oracle
from dpnl.core.symbolic import Symbolic, Input
from dpnl.core.variable import unknown


class EnumerationOracle(Oracle):
    """
    Automatically builds an oracle for a symbolic function S.
    Executes S(X) with the current instantiation:
    - Returns True if S(X) == S_output
    - Returns False if S(X) != S_output
    - Returns Unknown if S accesses an undefined RndVar.value
    """

    def __init__(self, S: Symbolic):
        super().__init__(S)
        self.root_cause_variable = None

    def __call__(self, I: Input, S_output):
        self.root_cause_variable = None
        for var in I:
            if not var.defined():
                self.root_cause_variable = var
                return unknown
        return self.S(I) == S_output

    def choose_variable_heuristic(self, X, S_output):
        """
        A simple variable selection heuristic:
        - Directly uses the variable that caused the oracle to return Unknown
        - Assumes this is the most informative branching point
        """
        return self.root_cause_variable


