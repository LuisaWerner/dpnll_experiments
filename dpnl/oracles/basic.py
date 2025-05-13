# === Simple automatic oracle derivation ===
#
# This method is a generalization of the method used to obtain the MNIST-N-SUM oracle from the addition.
#
# The main idea is that every symbolic function S (even a complex one) can be naturally programmed in python.
# Moreover, an implementation of S tends to be efficient, especially it tries to return a result as soon as possible.
# This often involve that only a part of the input is actually read by S. Thus, we can just execute S on partially
# valuated input, if S need to read a non valuated variable the oracle return unknown and if S terminates without
# reading a non valuated variable the oracle compare the actual result with the expected one. This oracle takes
# advantage of two things :
#   1. its complexity is equal to the complexity of the symbolic function which is especially interesting when the task
#      is complex but has an efficient algorithm to program S
#   2. the more efficient the S function is the more the oracle tends to have pruning capabilities (indeed it is not
#      systematic because S might be efficient but still need to read the whole inputs). A lot of concrete example of
#      S implementation produce a oracle with good pruning capabilities : the mnist-n-sum symbolic function based on
#      the base 10 addition, the graph reachability algorithm (useful for PNL tasks such as CiteSeer, Cora), every
#      efficient parsing algorithm tends to return as soon as the read prefix of the input is detected invalid (useful
#      if one want to use PNL to do grammar LLM aligned decoding for example)...
#
# Despite not being applicable in every situation, this automatic oracle derivation method is however very simple to
# implement for every function S and works well for numerous practical applications.
from contextlib import contextmanager
from typing import Any

from dpnl.core.oracle import Oracle
from dpnl.core.symbolic import Symbolic, Input
from dpnl.core.variable import Unknown, RndVar, unknown


class RndVarAccessError(Exception):
    """
    Raised when a symbolic function tries to access an undefined RndVar.value.
    Used to detect partial evaluations.
    """

    def __init__(self, var):
        self.var = var
        super().__init__(f"Unknown value of RndVar '{var.name}'")


@contextmanager
def temporary_getattribute(cls, new_method):
    """
    Temporarily monkey-patches cls.__getattribute__ to redirect attribute access.
    Used to trap reads of RndVar.value and detect access to unknowns.
    """
    original_method = cls.__getattribute__
    cls.__getattribute__ = new_method
    try:
        yield
    finally:
        cls.__getattribute__ = original_method


def wrapper_getattribute(self, name):
    """
    Custom __getattribute__ for RndVar:
    - Raises RndVarAccessError if .value is accessed while undefined
    """
    if name == "value":
        val = object.__getattribute__(self, name)
        if isinstance(val, Unknown):
            raise RndVarAccessError(self)
        return val
    return object.__getattribute__(self, name)


class BasicOracle(Oracle):
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

    def __call__(self, I: Input, S_output: Any):
        self.root_cause_variable = None
        with temporary_getattribute(RndVar, wrapper_getattribute):
            try:
                return self.S(I) == S_output
            except RndVarAccessError as e:
                self.root_cause_variable = e.var
                return unknown

    def choose_variable_heuristic(self, X, S_output):
        """
        A simple variable selection heuristic:
        - Directly uses the variable that caused the oracle to return Unknown
        - Assumes this is the most informative branching point
        """
        return self.root_cause_variable
