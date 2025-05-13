from dpnl.core.caching import CachingHash
from dpnl.core.symbolic import Input
from dpnl.logics.sat_logic import SATLogicSymbolic


def unit_propagation(clauses: list[list[int]], do_not_touch: set):
    while True:

        literals = set()
        for clause in clauses:
            if len(clause) == 0:
                return [[]]
            if len(clause) == 1:
                lit = clause[0]
                if lit not in do_not_touch and -lit not in do_not_touch:
                    if -lit in literals:
                        return [[]]
                    literals.add(lit)

        if len(literals) == 0:
            break

        new_clauses = []
        for clause in clauses:
            new_clause = []
            for lit in clause:
                if lit in literals:
                    new_clause = None
                    break
                if -lit not in literals:
                    new_clause.append(lit)
            if new_clause is not None:
                new_clauses.append(new_clause)

        clauses = new_clauses

    return clauses


def to_string(clauses: list[list[int]]):
    str_clauses = set(repr(sorted(set(clause))) for clause in clauses)
    return repr(sorted(str_clauses))


class SATCachingHash(CachingHash):
    """
    Caching hash based on the SAT logics. This caching hash function takes inspiration from the component caching
    in DPLL. The idea is that the SATLogicSymbolic function S given an input I produces a SAT which models corresponds
    to the sub-valuations of I such that the function return True. Therefore, if we use unit-propagation to simplify
    the SAT instance, two different I : I1 and I2 may lead to the same simplified SAT instance. Thus, if I1 and I2 have
    the same undefined variables it means that for all I1' sub-valuation of I1 and I2' sub-valuation of I2
    S(I1') = S(I2'). Therefore, SATCachingHash is both useful because it has collisions and is correct.
    """
    def __init__(self, S: SATLogicSymbolic):
        super().__init__(S)

    def __call__(self, I: Input):
        assert isinstance(self.S, SATLogicSymbolic)  # For pycharm typechecking
        literals = self.S.assumptions(I)
        literals.append(self.S.conclusion(I))
        clauses = [[lit.idx for lit in clause.literals] for clause in self.S.axioms]
        clauses.extend([[lit.idx] for lit in literals])
        do_not_touch = set()
        for var in I:
            if not var.defined():
                do_not_touch.update([lit.idx for lit in self.S.assumptions_linked_to(var)])
        function_state_hash = to_string(unit_propagation(clauses, do_not_touch))
        return function_state_hash
