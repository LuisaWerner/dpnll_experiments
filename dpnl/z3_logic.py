from typing import Any

import z3
from dpnl import Logic, unknown, LogicS


class Z3Logic(Logic):
    """
    Interface with the Z3 SMT solver
    """

    def __init__(self):
        super().__init__()

    def Not(self, formula):
        return z3.Not(formula)

    def Prove(self, assumptions: list[Any], conclusion: Any, vars2capture: tuple = ()):
        s = z3.Solver()
        s.reset()
        s.add(z3.Not(conclusion))
        check = str(s.check(*assumptions))
        if check == "unsat":
            # There is a proof
            core = s.unsat_core()
            return True, tuple(var in core for var in vars2capture)
        elif check == "sat":
            # There is no proof
            model = s.model()  # Get the counter example model

            def convert2bool(val):
                if val is None:
                    return unknown
                try:
                    return bool(val)
                except z3.z3types.Z3Exception:
                    return unknown

            return False, tuple(convert2bool(model[var]) for var in vars2capture)

        else:
            assert False, "Should not be possible if the logic symbolic function is well defined"


# As we can see by comparing the following function with the OptimizedGraphReachabilityOracle in graph_reachability.py,
# sometimes encoding the symbolic function in logic is as hard as designing a custom oracle optimized for the symbolic
# function.

def graph_reachability_S(N: int, src: int, dst: int):
    """
    Create a symbolic function for the graph reachability problem for a fixed size of graph and determined src and dst.
    :param N: The number of node of the graph
    :param src: The source node
    :param dst: The end node to reach from the source node
    :return: A graph reachability symbolic logic function for graphs of size N with determined src and dst
    """

    # Creating the variables
    edge = [[z3.Bool(f"edge({i},{j})") for j in range(N)] for i in range(N)]
    reached = [z3.Bool(f"reached({j})") for j in range(N)]

    # Inputs variables of the function
    inputs = ()
    for i in range(N):
        for j in range(N):
            inputs += (edge[i][j],)

    # Axioms
    axioms = [reached[src]]
    for i in range(N):
        for j in range(N):
            if i != j:
                axioms.append(z3.Implies(z3.And(reached[i], edge[i][j]), reached[j]))

    # Since the logic function is monotone, we can make use of negation by failure of proof and forgot this part
    """for k in range(N-1):
        for j in range(N):
            if j != src:
                axioms.append(z3.Implies(reached[k+1][j], z3.Or(*[z3.And(reached[k][i], edge[i][j]) for i in range(N)])))"""

    # Query
    query = reached[dst]

    return LogicS(
        logic=Z3Logic(),
        axioms=axioms,
        inputs_tuple=inputs,
        query=query
    )
