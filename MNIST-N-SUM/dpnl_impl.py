from typing import Any

from dpnl.core.symbolic import Input, Symbolic
from dpnl.core.variable import RndVar, unknown
from dpnl.core.problem import PNLProblem
from dpnl.oracles.enumeration import EnumerationOracle
from dpnl.oracles.basic import BasicOracle
from dpnl.oracles.logic import LogicOracle
from dpnl.logics.sat_logic import SATLogic, SATLogicSymbolic
from dpnl.cachings.sat_logic_hash import SATCachingHash


class MNISTInput(Input):
    def __init__(self, length: int, result: int):
        assert 0 <= result < 2 * 10 ** length
        self.num = [
            [RndVar(("num", 0, idx), {value: 0.1 for value in range(10)}) for idx in range(length)],
            [RndVar(("num", 1, idx), {value: 0.1 for value in range(10)}) for idx in range(length)]
        ]
        self.result = result
        self.length = length
        super().__init__(probabilistic_attributes={"num"})


class MNISTSymbolic(Symbolic):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __call__(self, I: MNISTInput):
        carry = 0
        r = I.result
        for i in range(I.length):
            idx = I.length - i - 1
            d = I.num[0][idx].value + I.num[1][idx].value + carry
            if d % 10 != r % 10:
                return False
            r = int(r / 10)
            carry = int(d / 10)
        return carry == r


class S2(Symbolic):
    """
    More reliable symbolic function because it is algorithmically
    simpler. The aim is to combine it with the enumeration oracle to have a
    ground truth reference for the probabilities results.
    """
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def __call__(self, I: MNISTInput):
        n1 = int(''.join([str(var.value) for var in I.num[0]]))
        n2 = int(''.join([str(var.value) for var in I.num[1]]))
        return n1 + n2 == I.result


class MNISTLogicSymbolic(SATLogicSymbolic):

    def __init__(self, length: int):
        super().__init__([])
        assert isinstance(self.logic, SATLogic)  # For pycharm typechecking

        # Setting internal variables
        self.length = length

        # Setting the inputs variables (inversed)
        self.num = [
            [self.var(("num", 0, idx), range(10)) for idx in range(length)],
            [self.var(("num", 1, idx), range(10)) for idx in range(length)]
        ]

        # Program
        self.r = [self.var(("r", idx), range(10)) for idx in range(length + 1)]
        self.c = [self.var(("c", idx), range(2)) for idx in range(length + 1)]
        self.clause([self.c[0] == 0])

        addition_func = lambda x, y, carry: (x + y + carry) % 10
        carry_func = lambda x, y, carry: int((x + y + carry) / 10)

        for idx in range(length):
            self.r[idx].equal(addition_func, self.num[0][idx], self.num[1][idx], self.c[idx])
            self.c[idx + 1].equal(carry_func, self.num[0][idx], self.num[1][idx], self.c[idx])
        self.r[length].equal(lambda x: x, self.c[length])

        # Variables and clauses to return the output value in function of the result we want to obtain
        self.output = self.var("output", [False, True])
        self.result_clause = self.logic.clause([])
        self.conv_result_clauses = [self.logic.clause([]) for _ in range(length + 1)]
        self.axioms.append(self.result_clause)
        self.axioms.extend(self.conv_result_clauses)

    def _set_result(self, result: int):
        # We change the result clause such that it is "self.r == result => self.output == True"
        new_result_clause = []
        for idx in range(self.length + 1):
            new_result_clause.append(-(self.r[idx] == result % 10))
            self.conv_result_clauses[idx].literals = [-(self.output == True), (self.r[idx] == result % 10)]
            result = int(result / 10)
        new_result_clause.append(self.output == True)
        self.result_clause.literals = new_result_clause

    def assumptions_linked_to(self, var: RndVar) -> list[Any]:
        assert isinstance(var.name, tuple) and len(var.name) == 3 and var.name[0] == "num"
        digit_var = self.num[var.name[1]][self.length - var.name[2] - 1]
        return list(digit_var.literals())

    def assumptions(self, I: MNISTInput) -> list[Any]:
        self._set_result(I.result)
        assumptions = []
        for i in range(2):
            for idx in range(self.length):
                var = I.num[i][idx]
                if var.defined():
                    digit_var = self.num[i][self.length - idx - 1]
                    assumptions.append(digit_var == var.value)

        return assumptions

    def conclusion(self, X: tuple[list[RndVar], list[RndVar], int]) -> Any:
        return self.output == True


# PNL problem

def problem(I: MNISTInput):
    return PNLProblem(I, MNISTSymbolic(I.length))


# Oracles

def enumeration(length: int):
    return EnumerationOracle(S2(length))


def basic(length: int):
    return BasicOracle(MNISTSymbolic(length))


def logic(length: int):
    return LogicOracle(MNISTLogicSymbolic(length))


# Caching Hash

def sat_logic_hash(length: int):
    return SATCachingHash(MNISTLogicSymbolic(length))
