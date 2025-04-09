from contextlib import contextmanager
from typing import Callable, Any
import abc


# === DPNL Components and Oracle Illustration ===
#
# This code illustrates how to implement Probabilistic NeuroSymbolic Logic (DPNL)
# using core Python components. It allows inference over symbolic functions S
# applied to uncertain inputs X made of probabilistic variables.
#
# Main features:
# - RndVar: Discrete probabilistic variables
# - PNLProblem: Encapsulates an input X and symbolic function S
# - basic_oracle: Automatically builds oracles from symbolic functions
# - dpnl(): Recursively computes the probability S(X) == output using dynamic evaluation


# === Core Representation ===

class Unknown:
    """
    Marker object to represent unknown/undefined values.
    Used as the initial value of random variables before they are instantiated.
    Can carry additional optional metadata for explanation purposes.
    """

    def __init__(self, optional=None):
        self.optional = optional

    def __repr__(self):
        return 'unknown'


unknown = Unknown()  # Global singleton for unknown value


class RndVar:
    """
    A discrete random variable with a domain and probability distribution.

    Attributes:
    - name: Identifier
    - domain_distrib: A dictionary mapping values (e.g. True/False) to probabilities
    - value: Current assigned value, or `unknown` if unassigned
    """

    def __init__(self, name: str, domain_distrib: dict, value: Any = None):
        assert 0.999 <= sum(domain_distrib.values()) <= 1.001  # Ensure probabilities sum to ~1
        self.name = name
        self.domain_distrib = domain_distrib
        self.value = value if value is not None else unknown

    def defined(self):
        """Returns True if the variable has a concrete (non-unknown) value."""
        return not isinstance(object.__getattribute__(self, "value"), Unknown)

    def domain(self):
        """Returns the set of all possible values this variable can take."""
        return set(self.domain_distrib)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return isinstance(other, RndVar) and id(self) == id(other)


class BoolRndVar(RndVar):
    """
    A convenient subclass of RndVar for binary (Boolean) variables.
    Uses a Bernoulli distribution with probability p for True.
    """

    def __init__(self, name: str, p: float, value: Any = None):
        super().__init__(name, {True: p, False: 1.0 - p}, value=value)


class RndVarIter:
    """
    Utility to extract all RndVar instances from structured input X.
    Traverses nested containers (like lists of lists) recursively.
    """

    def __init__(self, container: Any):
        self._array = []
        self._collect_rnd_vars(container)

    def _collect_rnd_vars(self, container: Any):
        if isinstance(container, RndVar):
            self._array.append(container)
        else:
            try:
                for sub_container in iter(container):
                    self._collect_rnd_vars(sub_container)
            except TypeError:
                pass  # Ignore non-iterables

    def __iter__(self):
        return iter(self._array)


# === Probabilistic Neurosymbolic Logic Problem ===

class PNLProblem:
    """
    Represents a DPNL problem: evaluating the probability that a symbolic function S
    returns a given output when applied to a probabilistic input X.

    Attributes:
    - X: Structured input (e.g., a graph), possibly containing RndVars
    - S: A symbolic function S(X) → output
    """

    def __init__(self, X: Any, S: Callable):
        self.X = X
        self.rnd_vars = RndVarIter(X)
        self.S = S

    def setX(self, X: Any):
        """Update the input X and its associated random variables."""
        self.X = X
        self.rnd_vars = RndVarIter(X)

    def prob(self, S_oracle: Callable, S_output: Any, choose: Callable = None):
        """
        Computes the probability that S(X) == S_output using recursive inference.
        Uses an oracle to shortcut evaluation when possible.

        Args:
        - S_oracle: Function to check whether S(X) == S_output (returns True/False/Unknown)
        - S_output: Desired output value to match
        - choose: Heuristic to select the next RndVar to instantiate (optional)

        Returns:
        - A float between 0 and 1 representing the probability
        """
        if choose is None:
            # Default: pick first undefined variable
            def blind_heuristic(pnl_instance: PNLProblem, S_output, oracle_answer):
                for X_var in self.rnd_vars:
                    if not X_var.defined():
                        return X_var

            choose = blind_heuristic

        def dpnl():
            oracle_answer = S_oracle(self.X, S_output)
            if oracle_answer is True:
                return 1.0
            elif oracle_answer is False:
                return 0.0
            else:
                # Branch over an undefined variable
                assert isinstance(oracle_answer, Unknown)
                rnd_var = choose(self, S_output, oracle_answer)
                result = 0.0
                for val, prob in rnd_var.domain_distrib.items():
                    rnd_var.value = val
                    result += prob * dpnl()
                rnd_var.value = unknown  # backtrack after recursion
                return result

        return dpnl()


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


def basic_oracle(S: Callable):
    """
    Automatically builds an oracle for a symbolic function S.
    Evaluates S(X) and:
    - Returns True if result == expected
    - Returns False if result ≠ expected
    - Returns Unknown if any RndVar is accessed while undefined
    """

    def S_oracle(X, o):
        with temporary_getattribute(RndVar, wrapper_getattribute):
            try:
                return S(X) == o
            except RndVarAccessError as e:
                return Unknown(optional={'root_cause_var': e.var})

    return S_oracle


def basic_oracle_choose_heuristic(pnl_problem: PNLProblem, S_output, oracle_answer: Unknown):
    """
    A simple variable selection heuristic:
    - Directly uses the variable that caused the oracle to return Unknown
    - Assumes this is the most informative branching point
    """
    return oracle_answer.optional['root_cause_var']


# === Logic oracles ===
#
# This part is an implementation of the general logic oracle described in Algorithm 3.
# It provides an abstract python class to provide an interface with numerous non-quantified logics : AbstractLogic
# It provides a python class to represent the symbolic function S in such a logic : LogicS
# It provides a function to automatically derive an oracle from a LogicS instance : AbstractLogic.Oracle
#

class AbstractLogic(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def Not(self, formula):
        pass

    @abc.abstractmethod
    def CheckProof(self, assumptions: list[Any], conclusion: Any):
        pass

    def Oracle(self, S):
        """
        This function automatically derive an oracle for S using Algorithm 3.
        :param S: A logic symbolic function
        :return: A logic oracle for S
        """

        assert isinstance(S, LogicS) and isinstance(S.logic, type(self))

        def oracle(X, S_output: bool):
            assumptions = []
            for i in range(len(S.inputs_tuple)):
                if X[i].defined():
                    if X[i].value:
                        assumptions.append(S.inputs_tuple[i])
                    else:
                        assumptions.append(S.logic.Not(S.inputs_tuple[i]))

            res = unknown
            pos_proof = S.logic.CheckProof(S.axioms + assumptions, S.query)
            neg_proof = S.logic.CheckProof(S.axioms + assumptions, S.logic.Not(S.query))

            if pos_proof:
                res = True
            elif neg_proof:
                res = False

            if not isinstance(res, Unknown) and not S_output:
                res = not res

            return res

        return oracle


class LogicS:
    def __init__(self, logic: AbstractLogic, axioms: list[Any], inputs_tuple: tuple, query: Any):
        """
        The class define a symbolic function S which considering a valuations of the logic formulas of inputs_tuple
        and the axioms return True if and only if there is a proof of query.
        :param logic: The logic in which it is expressed
        :param axioms: The list of axioms formulas that represents the knowledge of symbolic function context
        :param inputs_tuple: The tuple of logic formulas or atoms that constitutes the inputs of the function
        :param query: The formula that represent the meaning of the function
        """
        self.logic = logic
        self.axioms = axioms
        self.inputs_tuple = inputs_tuple
        self.query = query

    def __call__(self, X):
        assert isinstance(X, tuple) and len(X) == len(self.inputs_tuple)
        assumptions = []
        for i in range(len(self.inputs_tuple)):
            if X[i].value:
                assumptions.append(self.inputs_tuple[i])
            else:
                assumptions.append(self.logic.Not(self.inputs_tuple[i]))
        return self.logic.CheckProof(self.axioms + assumptions, self.query)



