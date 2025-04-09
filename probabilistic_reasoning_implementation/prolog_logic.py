import subprocess
from dpnl import PNLProblem, AbstractLogic, LogicS, Unknown, unknown, BoolRndVar
import json
from prolog_solver import Fact, Rule, Term, solve_clauses


def CheckProofTraced(assumptions: list, conclusion) -> tuple[bool, list[int]]:
    result = solve_clauses(assumptions, conclusion)
    if result is None:
        return False, []
    else:
        return True, list(result)


class PrologLogic(AbstractLogic):
    def __init__(self):
        super().__init__()

    def And(self, *formulas):
        return ", ".join(formulas)

    def Not(self, formula):
        return f"\\+({formula})"

    def CheckProof(self, assumptions: list[str], conclusion: str) -> bool:
        return CheckProofTraced(assumptions, conclusion)[0]

    def Oracle(self, S):
        """
        Because of the negation by failure in prolog which semantics is different form the real negation,
        the behavior of the general Oracle defined in AbstractLogic is wrong, we have to modify things a bit.
        :param S:
        :return:
        """

        assert isinstance(S, LogicS) and isinstance(S.logic, PrologLogic)

        def oracle(X, S_output: bool):
            # Here to negate something we just have not to put it inside the assumptions, with negation by failure
            # it will count as a negation
            unknown2true_assumptions = []
            unknown2true_variables = []
            unknown2false_assumptions = []
            for i in range(len(S.inputs_tuple)):
                if X[i].defined():
                    if X[i].value:
                        unknown2true_assumptions.append(S.inputs_tuple[i])
                        unknown2true_variables.append(X[i])
                        unknown2false_assumptions.append(S.inputs_tuple[i])
                else:
                    unknown2true_assumptions.append(S.inputs_tuple[i])
                    unknown2true_variables.append(X[i])

            unknown2true_proof, used_assumptions = CheckProofTraced(S.axioms + unknown2true_assumptions, S.query)
            unknown2false_proof = S.logic.CheckProof(S.axioms + unknown2false_assumptions, S.query)
            used_variables = [idx - len(S.axioms) for idx in used_assumptions if len(S.axioms) <= idx]

            if unknown2false_proof:
                # If there is already a proof only considering assumptions valuated to true we can return true
                res = True
            elif not unknown2true_proof:
                # If there is no proof even when considering all unknown assumptions as true we can return false
                res = False
            else:
                root_cause = None
                for idx in used_variables:
                    if not unknown2true_variables[idx].defined():
                        root_cause = unknown2true_variables[idx]
                assert root_cause is not None, "Not possible except for a bug"
                res = Unknown(optional=root_cause)

            if not isinstance(res, Unknown) and not S_output:
                res = not res

            return res

        return oracle


def choose_heuristic(pnl_instance: PNLProblem, S_output, oracle_answer):
    return oracle_answer.optional


def graph_reachability_S(N: int, src: int, dst: int):
    """
    Create a symbolic function for the graph reachability problem for a fixed size of graph and determined src and dst.
    :param N: The number of node of the graph
    :param src: The source node
    :param dst: The end node to reach from the source node
    :return: A graph reachability symbolic logic function for graphs of size N with determined src and dst
    """

    inputs = ()
    for i in range(N):
        for j in range(N):
                inputs += (Fact(Term("edge", [Term(str(i)), Term(str(j))])),)

    # Reachability rules
    rules = [
        Rule(Term("reachable", [Term("X"), Term("Y")]),
             [Term("edge", [Term("X"), Term("Y")])]),
        Rule(Term("reachable", [Term("X"), Term("Y")]),
             [Term("edge", [Term("X"), Term("Z")]), Term("reachable", [Term("Z"), Term("Y")])])
    ]

    query = [Term("reachable", [Term(str(src)), Term(str(dst))])]

    return LogicS(
        logic=PrologLogic(),
        axioms=rules,
        inputs_tuple=inputs,
        query=query
    )

"""N = 7
X = ()
for i in range(N):
    for j in range(N):
        X += (BoolRndVar("", 0.5),)
pnl_problem = PNLProblem(X, graph_reachability_S(N, 0, 1))
oracle = pnl_problem.S.logic.Oracle(pnl_problem.S)
choose = choose_heuristic
pnl_problem.prob(oracle, True, choose)"""