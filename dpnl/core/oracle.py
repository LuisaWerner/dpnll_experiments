from contextlib import contextmanager
from typing import Callable, Any
import abc

from dpnl.core.symbolic import Symbolic, Input
from dpnl.core.variable import RndVar, unknown, Unknown


# === Oracle instance ===

class Oracle(abc.ABC):
    """
    Abstract base class representing an oracle. An oracle answers whether
    S(I) == S_output for all, none, or some instantiations of the random variables in I.
    Also defines a heuristic to select which variable to instantiate next.
    """

    def __init__(self, S: Symbolic):
        self.S = S

    @abc.abstractmethod
    def __call__(self, I: Input, S_output: Any):
        """
        The actual oracle algorithm
        :param I: Structured input of S (e.g., a graph), containing RndVars
        :param S_output: The desired output of S
        :return: True iif for all valuations of the RndVars in X, S(X)=S_output, False iif for all valuations of the
        RndVars in I, S(I)!=S_output and an instance of unknown otherwise.
        """
        pass

    def choose_variable_heuristic(self, I: Input, S_output: Any):
        """
        This method return a variable of X that is unknown to do the branching in DPNL. By default, it is the
        blind heuristic who just choose the first unknown valuated variable it finds in X. But this method is meant
        to be overwritten.
        :param I: Structured input of S (e.g., a graph), containing RndVars
        :param S_output: The desired output of S
        :return: A RndVar v of I such that v.value = unknown.
        """
        for v in I:
            if not v.defined():
                return v
        return None
