# === Logic oracles ===
#
# This part is an implementation of the general logic oracle described in Algorithm 3.
# It provides an abstract python class to provide an interface with numerous non-quantified logics : AbstractLogic
# It provides a python class to represent the symbolic function S in such a logic : LogicS
# It provides a function to automatically derive an oracle from a LogicS instance : AbstractLogic.Oracle
#
import abc
import random
from typing import Any

from dpnl.core.oracle import Oracle
from dpnl.core.symbolic import Symbolic, Input
from dpnl.core.variable import unknown, RndVar


class Model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def truth_value(self, assumption: Any) -> bool | type(None):
        pass


class Logic(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def neg(self, formula):
        pass

    @abc.abstractmethod
    def model(self, axioms: list[Any], assumptions: list[Any]) -> Model | type(None):
        """
        The solver main method.
        :param axioms: The list of axioms formulas of the logics
        :param assumptions: The list of assumptions formulas of the logics
        :return:
            if satisfiable(axioms and assumptions):
                return the corresponding model
            else:
                return None
        """
        pass

    #@abc.abstractmethod
    #def simplify(self, axioms: list[Any], assumptions: list[Any]) -> list[Any]:
        """
        This method is intuitively the generalization of the unit propagation for SAT. It simplifies the formula
        'axioms and assumptions' to the maximum possible. If the simplification is equivalent to True or False it
        returns True or False.
        :param axioms: The list of axioms formulas of the logics
        :param assumptions: The list of assumptions formulas of the logics
        :return: True | False | simplification of 'axioms and assumptions'
        """
    #    pass


class LogicSymbolic(Symbolic):
    def __init__(self, logic: Logic, axioms: list[Any]):
        super().__init__()
        self.logic = logic
        self.axioms = axioms

    @abc.abstractmethod
    def assumptions_linked_to(self, var: RndVar) -> list[Any]:
        """
        :param var: A random variable related to the input of the logic function
        :return: The list of assumptions related to var
        """
        pass

    @abc.abstractmethod
    def assumptions(self, I: Input) -> list[Any]:
        """
        :param I: An input of the logic function
        :return: The list of assumptions corresponding to X
        """
        pass

    @abc.abstractmethod
    def conclusion(self, I: Input) -> Any:
        """
        :param I: An input of the logic function
        :return: The conclusion to prove corresponding to X
        """
        pass

    def __call__(self, I: Input):
        assumptions = self.assumptions(I)
        assumptions.append(self.logic.neg(self.conclusion(I)))
        return self.logic.model(self.axioms, assumptions) is None


class LogicOracle(Oracle):
    """
    Oracle implementation for symbolic functions defined via logic inference.
    Uses a prover to check whether assumptions and axioms entail the query.
    """

    def __init__(self, S: LogicSymbolic):
        super().__init__(S)

        # Cache information necessary to the heuristic to choose variable
        self.pos_model = None
        self.neg_model = None

    def __call__(self, I: Input, S_output: bool):
        assert isinstance(self.S, LogicSymbolic)  # Just for typechecking in pycharm

        # Initializing
        logic = self.S.logic
        axioms = self.S.axioms
        assumptions = self.S.assumptions(I)
        conclusion = self.S.conclusion(I)

        # Gathering models
        self.pos_model = logic.model(axioms, assumptions + [conclusion])
        self.neg_model = logic.model(axioms, assumptions + [logic.neg(conclusion)])

        # Computing the result based on the results of the models
        res = unknown
        if self.pos_model is None:  # It means there is a proof of "axioms and assumptions :- neg(conclusion))
            res = False
        elif self.neg_model is None:  # It means there is a proof of "axioms and assumptions :- conclusion)
            res = True
        if res is not unknown and S_output is False:  # We invert the result if necessary
            res = not res
        return res

    def choose_variable_heuristic(self, I: Input, S_output: bool):
        """
        This heuristic return the first variable that is unknown and has different or unknown value in the models
        because this variable is likely to be important in the decision process.
        """
        assert isinstance(self.S, LogicSymbolic)  # Just for typechecking in pycharm
        assert self.pos_model is not None and self.neg_model is not None

        possible_vars = []

        for var in I:
            if not var.defined():
                for assumption in self.S.assumptions_linked_to(var):
                    if self.pos_model.truth_value(assumption) != self.neg_model.truth_value(assumption):
                        possible_vars.append(var)

        assert len(possible_vars) > 0
        return possible_vars[0]
