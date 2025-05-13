from itertools import product
from typing import Any, Callable, Iterable

from dpnl.logics.sat_logic.formula import Literal, Clause
from dpnl.logics.sat_logic.core import SATLogic, SATModel
from dpnl.oracles.logic import LogicSymbolic


class VariableLiteral(Literal):
    def __init__(self, variable, specific_value, sat_logic: SATLogic):
        name = (variable.name, "==", specific_value)
        idx = sat_logic.count
        sat_logic.count += 1
        idx2lit = sat_logic.idx2lit
        super().__init__(name, idx, idx2lit)
        self._variable = variable

    def variable(self):
        return self._variable

    def specific_value(self):
        return self.name[2]


class Variable:
    def __init__(self, name, val2lit: dict[Any, Literal], logicS, neg: bool = False):
        assert isinstance(logicS, SATLogicSymbolic)
        self.name = name
        self.val2lit = val2lit
        self.logicS = logicS
        self.neg = neg

    def equal(self, func: Callable, *args):
        assert isinstance(self.logicS, SATLogicSymbolic)

        def rel_func(output, *inputs):
            return func(*inputs) == output

        self.logicS.clauses(
            relation=rel_func,
            clause_blueprint=[self] + [-var for var in args]
        )

    def literals(self):
        return list(self.val2lit.values())

    def domain(self):
        return list(self.val2lit.keys())

    def get_possible_values(self, model: SATModel):
        for val, lit in self.val2lit.items():
            if model.truth_value(lit) is None:
                return self.domain()
            elif model.truth_value(lit) is True:
                return [val]

    def __neg__(self):
        return Variable(self.name, self.val2lit, self.logicS, not self.neg)

    def __eq__(self, other) -> Literal:
        if self.neg:
            return -self.val2lit[other]
        else:
            return self.val2lit[other]

    def val(self, value) -> Literal:
        if self.neg:
            return -self.val2lit[value]
        else:
            return self.val2lit[value]


class SATLogicSymbolic(LogicSymbolic):
    def __init__(self, axioms: list[Clause]):
        logic = SATLogic()
        super().__init__(logic, axioms)

    def var(self, name, domain: Iterable):
        assert isinstance(self.logic, SATLogic)

        # Creating the variable
        domain = list(domain)
        multi_var = Variable(name, {}, self)
        val2lit = {
            val: VariableLiteral(multi_var, val, self.logic) for val in domain
        }
        multi_var.val2lit = val2lit

        # Adding the axioms on its literals
        self.axioms.append(self.logic.clause([multi_var == val for val in domain]))
        for i in range(len(domain)):
            for j in range(i + 1, len(domain)):
                self.axioms.append(self.logic.clause([-(multi_var == domain[i]), -(multi_var == domain[j])]))
        return multi_var

    def clause(self, literals: list[Literal]):
        assert isinstance(self.logic, SATLogic)
        self.axioms.append(
            self.logic.clause(literals)
        )

    def clauses(self, relation: Callable, clause_blueprint: list[Variable]):
        args = tuple(clause_blueprint)
        args_domain = tuple(list(arg.val2lit.keys()) for arg in args)
        all_inputs = list(product(*args_domain))
        for inputs in all_inputs:
            if relation(*inputs):
                self.clause([(args[i] == inputs[i]) for i in range(len(args))])

    def imply(self, conjunction: list[Literal], disjunction: list[Literal]):
        self.clause([-lit for lit in conjunction]+disjunction)

    def equiv(self, conjunction_1: list[Literal], conjunction_2: list[Literal]):
        disjunction_1 = [-lit for lit in conjunction_1]
        disjunction_2 = [-lit for lit in conjunction_2]
        for lit in conjunction_1:
            self.clause(disjunction_2 + [lit])
        for lit in conjunction_2:
            self.clause(disjunction_1 + [lit])

    def equivs(self, relation: Callable, conjunction_blueprint_1: list[Variable], conjunction_blueprint_2: list[Variable]):
        args1 = tuple(conjunction_blueprint_1)
        args2 = tuple(conjunction_blueprint_2)
        args1_domain = tuple(list(arg.val2lit.keys()) for arg in args1)
        args2_domain = tuple(list(arg.val2lit.keys()) for arg in args2)
        all_args1 = list(product(*args1_domain))
        all_args2 = list(product(*args2_domain))
        for values1 in all_args1:
            for values2 in all_args2:
                if relation(*(values1+values2)):
                    self.equiv(
                        [(args1[i] == values1[i]) for i in range(len(args1))],
                        [(args2[i] == values2[i]) for i in range(len(args2))]
                    )
