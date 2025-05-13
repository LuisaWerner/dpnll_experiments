from pysat.solvers import Minisat22

from dpnl.logics.sat_logic.formula import Literal, Clause, Conj
from dpnl.oracles.logic import Model, Logic

# SAT solver : We can choose the best suited solver for this task, Minisat22 works well on our setup
SolverImpl = Minisat22


class SATModel(Model):
    def __init__(self, logic, solver_model):
        super().__init__()
        self.logic = logic
        self.solver_model = set(self.logic.lit_of_idx(idx) for idx in solver_model)

    def truth_value(self, assumption: Literal) -> bool | type(None):
        if assumption in self.solver_model:
            return True
        if -assumption in self.solver_model:
            return False
        return None

    def __repr__(self):
        assert isinstance(self.logic, SATLogic)
        return repr(self.solver_model)


class SATLogic(Logic):
    """
    Interface with the Z3 SMT solver
    """

    def __init__(self):
        super().__init__()
        self.count = 1
        self.idx2lit = {}  # Inverse mapping for debug purpose
        self.solver_map = {}  # Caching of solver for efficiency

    def neg(self, formula):
        if isinstance(formula, Literal):
            return -formula
        if isinstance(formula, Clause):
            return Conj([-lit for lit in formula.literals])
        if isinstance(formula, Conj):
            return Clause([-lit for lit in formula.literals])
        assert 0

    def lit_of_idx(self, idx: int):
        return self.idx2lit.get(idx, -self.idx2lit.get(-idx, 0))

    def lit(self, name):
        idx = self.count
        self.count += 1
        return Literal(name, idx, self.idx2lit)

    def clause(self, literals: list[Literal]):
        return Clause(literals)

    def conj(self, literals: list[Literal]):
        return Conj([lit for lit in literals])

    def model(self, axioms: list[Clause], assumptions: list[Literal]) -> SATModel | type(None):

        # Caching solver
        solver = self.solver_map.get(
            id(axioms),
            SolverImpl(bootstrap_with=[clause.get_literals() for clause in axioms])
        )
        self.solver_map[id(axioms)] = solver

        # Calling the solver
        satisfiable = solver.solve(assumptions=[lit.idx for lit in assumptions])

        if satisfiable:
            return SATModel(self, solver.get_model())
        else:
            # Extract the unsatisfiable core
            core = solver.get_core()
            if core:
                # Learn a new clause to prevent the same conflict
                learned_clause = [-lit for lit in core]
                if len(learned_clause) < len(assumptions):
                    solver.add_clause(learned_clause)
            return None
