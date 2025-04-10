from contextlib import contextmanager
from typing import Callable, Any
import abc


# === DPNL Components and Oracle Illustration ===
#
# This code illustrates how to implement Dynamic Probabilistic NeuroSymbolic Logic (DPNL)
# using core Python components. It enables probabilistic inference over symbolic functions S
# applied to uncertain inputs X consisting of random variables.
#
# Key components:
# - RndVar: Represents discrete random variables
# - Oracle : The oracle abstract class
# - PNLProblem: Encapsulates a symbolic function S and its probabilistic input X
# - dpnl(): Recursively computes the probability that S(X) == target output using dynamic inference
# - BasicOracle: Automatically builds an oracle from any symbolic function S
# - LogicS : Class for symbolic functions represented by a logic program
# - LogicOracle: Oracle to automatically generate an oracle for a LogicS instance based on Algorithm 3 of the paper


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

    def __eq__(self, other):
        return isinstance(other, Unknown)


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


# === Oracle instance ===

class Oracle(abc.ABC):
    """
    Abstract base class representing an oracle. An oracle answers whether
    S(X) == S_output for all, none, or some instantiations of the random variables in X.
    Also defines a heuristic to select which variable to instantiate next.
    """

    def __init__(self, S: Callable):
        self.S = S

    @abc.abstractmethod
    def __call__(self, X, S_output):
        """
        The actual oracle algorithm
        :param X: Structured input of S (e.g., a graph), containing RndVars
        :param S_output: The desired output of S
        :return: True iif for all valuations of the RndVars in X, S(X)=S_output, False iif for all valuations of the
        RndVars in X, S(X)!=S_output and an instance of unknown otherwise.
        """
        pass

    def choose_variable_heuristic(self, X, S_output):
        """
        This method return a variable of X that is unknown to do the branching in DPNL. By default, it is the
        blind heuristic who just choose the first unknown valuated variable it finds in X. But this method is meant
        to be overwritten.
        :param X: Structured input of S (e.g., a graph), containing RndVars
        :param S_output: The desired output of S
        :return: A RndVar v of X such that v.value = unknown.
        """
        for v in RndVarIter(X):
            if not v.defined():
                return v
        return None


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

    def prob(self, S_oracle: Oracle, S_output: Any):
        """
        Computes the probability that S(X) == S_output via recursive evaluation.
        Leverages the oracle to avoid unnecessary branching when a definite answer is available.

        Args:
        - S_oracle: A valid Oracle for self.S
        - S_output: Desired output of self.S(X)

        Returns:
        - A float in [0, 1] representing the probability
        """

        def dpnl():
            oracle_answer = S_oracle(self.X, S_output)
            if oracle_answer is True:
                return 1.0
            elif oracle_answer is False:
                return 0.0
            else:
                # Branch over an undefined variable
                assert isinstance(oracle_answer, Unknown)
                rnd_var = S_oracle.choose_variable_heuristic(self.X, S_output)
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


class BasicOracle(Oracle):
    """
    Automatically builds an oracle for a symbolic function S.
    Executes S(X) with the current instantiation:
    - Returns True if S(X) == S_output
    - Returns False if S(X) != S_output
    - Returns Unknown if S accesses an undefined RndVar.value
    """
    def __init__(self, S: Callable):
        super().__init__(S)
        self.root_cause_variable = None

    def __call__(self, X, S_output):
        self.root_cause_variable = None
        with temporary_getattribute(RndVar, wrapper_getattribute):
            try:
                return self.S(X) == S_output
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


# === Logic oracles ===
#
# This part is an implementation of the general logic oracle described in Algorithm 3.
# It provides an abstract python class to provide an interface with numerous non-quantified logics : AbstractLogic
# It provides a python class to represent the symbolic function S in such a logic : LogicS
# It provides a function to automatically derive an oracle from a LogicS instance : AbstractLogic.Oracle
#

class Logic(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def Not(self, formula):
        pass

    @abc.abstractmethod
    def Prove(self, assumptions: list[Any], conclusion: Any, vars2capture: tuple = ()):
        """
        The solver main method.
        :param assumptions: The list of assumptions formulas of the logics
        :param conclusion: The conclusion to prove
        :param vars2capture: Additional tuple of variables of the logic
        :return:
            If assumptions :- conclusion return True, tuple(true if var is used in the proof for var in vars2capture)
            Else return False, tuple(value of var in the counter example model for var in vars2capture)
        """
        pass

    class LogicOracle(Oracle):
        """
        Oracle implementation for symbolic functions defined via logic inference.
        Supports non-monotonic logic with negation by failure.
        Uses a prover to check whether assumptions and axioms entail the query.
        """
        def __init__(self, S):
            assert isinstance(S, LogicS)
            super().__init__(S)
            self.pos_counter_model = None
            self.neg_counter_model = None

        def __call__(self, X: tuple[RndVar], S_output: bool):

            self.pos_counter_model = None
            self.neg_counter_model = None

            assert isinstance(self.S, LogicS)  # Just for typechecking in pycharm
            S = self.S
            logic = S.logic

            assumptions = []
            for i in range(len(S.inputs_tuple)):
                if X[i].defined():
                    if X[i].value:
                        assumptions.append(S.inputs_tuple[i])
                    else:
                        assumptions.append(S.logic.Not(S.inputs_tuple[i]))

            res = unknown
            pos_proof, pos_captured = logic.Prove(S.axioms + assumptions, S.query, vars2capture=S.inputs_tuple)
            neg_proof, neg_captured = logic.Prove(S.axioms + assumptions, logic.Not(S.query),
                                                  vars2capture=S.inputs_tuple)

            if pos_proof:
                res = True
            elif neg_proof:
                res = False
            else:
                self.pos_counter_model = pos_captured
                self.neg_counter_model = neg_captured

            if not isinstance(res, Unknown) and not S_output:
                res = not res

            return res

        def choose_variable_heuristic(self, X: tuple[RndVar], S_output: bool):
            """
            This heuristic return the first variable that is unknown and has different or unknown value in the models
            because this variable is likely to be important in the decision process.
            """
            for i in range(len(X)):
                if not X[i].defined() and (self.pos_counter_model[i] != self.neg_counter_model[i]):
                    return X[i]
            for i in range(len(X)):
                if not X[i].defined() and (
                        unknown == self.pos_counter_model[i] or unknown == self.neg_counter_model[i]):
                    return X[i]

    class LogicOracleMonotone(Oracle):
        """
        Oracle for monotonic logical functions, where increasing truth values (False → True)
        never decreases the output (i.e., S remains True or becomes True).

        This allows pruning based on evaluating S under extreme valuations. This oracle is really useful because  it is
        very efficient and every deterministic function S running in polynomial time can be coded in Horn-SAT by
        Cook-Levin-like transformation making the corresponding logic symbolic function monotone.
        """
        def __init__(self, S):
            assert isinstance(S, LogicS)
            super().__init__(S)
            self.used_vars = None

        def __call__(self, X: tuple[RndVar], S_output: bool):

            self.used_vars = None

            assert isinstance(self.S, LogicS)  # Just for typechecking in pycharm
            S = self.S
            logic = S.logic

            unknown2true_assumptions = []
            unknown2false_assumptions = []
            for i in range(len(S.inputs_tuple)):
                if X[i].defined():
                    if X[i].value:
                        unknown2true_assumptions.append(S.inputs_tuple[i])
                        unknown2false_assumptions.append(S.inputs_tuple[i])
                    else:
                        unknown2true_assumptions.append(logic.Not(S.inputs_tuple[i]))
                        unknown2false_assumptions.append(logic.Not(S.inputs_tuple[i]))
                else:
                    unknown2true_assumptions.append(S.inputs_tuple[i])
                    unknown2false_assumptions.append(logic.Not(S.inputs_tuple[i]))

            res = unknown
            unknown2true_proof, used_vars = logic.Prove(S.axioms + unknown2true_assumptions, S.query,
                                                        vars2capture=S.inputs_tuple)
            unknown2false_proof, _ = logic.Prove(S.axioms + unknown2false_assumptions, S.query)

            if unknown2false_proof:
                # It means that even if we put all unknown instance to false S return true
                res = True
            if not unknown2true_proof:
                # It means that even if we all unknown instance to true S return false
                res = False
            else:
                self.used_vars = used_vars

            if not unknown == res and not S_output:
                res = not res

            return res

        def choose_variable_heuristic(self, X: tuple[RndVar], S_output: bool):
            """
            This heuristic choose the first unknown variable that is used in the proof of conclusion when considering
            every unknown variable as true.
            """
            for i in range(len(X)):
                if not X[i].defined() and self.used_vars[i]:
                    return X[i]


class LogicS:
    """
    Represents a symbolic function S expressed in a formal logic system.
    S(X) returns True iff the axioms and current assumptions (based on X)
    logically entail the query.

    Args:
    - logic: The logic system in which S is defined
    - axioms: Static knowledge base
    - inputs_tuple: Logical atoms or formulas controlled by probabilistic variables
    - query: The logical statement to verify
    """
    def __init__(self, logic: Logic, axioms: list[Any], inputs_tuple: tuple, query: Any):
        self.logic = logic
        self.axioms = axioms
        self.inputs_tuple = inputs_tuple
        self.query = query

    def __call__(self, X: tuple[RndVar]):
        assert isinstance(X, tuple) and len(X) == len(self.inputs_tuple)
        assumptions = []
        for i in range(len(self.inputs_tuple)):
            if X[i].value:
                assumptions.append(self.inputs_tuple[i])
            else:
                assumptions.append(self.logic.Not(self.inputs_tuple[i]))
        return self.logic.Prove(self.axioms + assumptions, self.query)[0]
