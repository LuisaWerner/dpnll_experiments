from typing import Any, Callable

from dpnl.core.caching import CachingHash
from dpnl.core.oracle import Oracle
from dpnl.core.symbolic import Input, Symbolic
from dpnl.core.variable import unknown, Unknown


# === Probabilistic Neurosymbolic Logic Problem ===

class PNLProblem:
    """
    Represents a DPNL problem: evaluating the probability that a symbolic function S
    returns a given output when applied to a probabilistic input X.

    Attributes:
    - X: Structured input (e.g., a graph), possibly containing RndVars
    - S: A symbolic function S(X) → output
    """

    def __init__(self, I: Input, S: Symbolic):
        self.I = I
        self.S = S

        # temporary attributes
        self._cache = {}
        self._caching_hash = None
        self._oracle = None
        self._output = None

    def _clear(self):
        self._cache = {}
        self._caching_hash = None
        self._oracle = None
        self._output = None

    def sample_prob(self, S_output: Any, samples: int):
        count = 0
        unknown_vars = [var for var in self.I if not var.defined()]
        for sample in range(samples):
            for var in unknown_vars:
                var.value = var.sample()
            if self.S(self.I) == S_output:
                count += 1
        for var in unknown_vars:
            var.value = unknown
        return float(count) / samples

    def DPNL(self):

        oracle_answer = self._oracle(self.I, self._output)
        if oracle_answer is True:
            return 1.0
        elif oracle_answer is False:
            return 0.0
        else:
            # Branch over an undefined variable
            assert oracle_answer is unknown
            rnd_var = self._oracle.choose_variable_heuristic(self.I, self._output)
            result = 0.0
            for val, prob in rnd_var.domain_distrib.items():
                rnd_var.value = val
                result += prob * self.DPNL()
            rnd_var.value = unknown  # backtrack after recursion
            return result

    def DPNLCache(self):
        oracle_answer = self._oracle(self.I, self._output)
        if oracle_answer is True:
            return 1.0
        elif oracle_answer is False:
            return 0.0
        else:


            # Branch over an undefined variable
            assert oracle_answer is unknown

            # Try to hit the cache
            h = self._caching_hash(self.I)
            try:
                return self._cache[h]
            except KeyError:
                pass

            # Cache miss we must compute the probability
            rnd_var = self._oracle.choose_variable_heuristic(self.I, self._output)
            result = 0.0
            for val, prob in rnd_var.domain_distrib.items():
                rnd_var.value = val
                result += prob * self.DPNLCache()
            rnd_var.value = unknown  # backtrack after recursion
            self._cache[h] = result  # Saving to cache
            return result

    def Proba(self, output: Any, oracle: Oracle, caching_hash: CachingHash = None):
        """
        Computes the probability that self.S(self.I) == output via recursive evaluation.
        Leverages the oracle to avoid unnecessary branching when a definite answer is available.

        Args:
        - output: Desired output of self.S(I)
        - oracle: A valid Oracle for self.S


        Returns:
        - A float in [0, 1] representing the probability
        """

        self._output = output
        self._oracle = oracle
        self._cache = {}
        self._caching_hash = caching_hash

        if caching_hash is not None:
            return self.DPNLCache()
        else:
            return self.DPNL()
