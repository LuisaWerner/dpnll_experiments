import z3
from dpnl import AbstractLogic, LogicS


class Z3Logic(AbstractLogic):

    def __init__(self):
        super().__init__()

    def Not(self, formula):
        return z3.Not(formula)

    def CheckProof(self, assumptions: list, conclusion):
        s = z3.Solver()
        for formula in assumptions:
            s.add(formula)
        return str(s.check(z3.Not(conclusion))) == "unsat"


# Example of definition of Z3 LogicS symbolic function : self-driving car example

# Inputs logic variables
obstacle_ahead = z3.Bool('obstacle_ahead')
brake_ok = z3.Bool('brake_ok')
road_slippery = z3.Bool('road_slippery')
self_speed = z3.Int('self_speed')  # We suppose that it takes a finite range of value between 0 and 100 for example
dist_obstacle = z3.Int('dist_obstacle')  # We suppose that it takes a finite range of value between 0 and 100 for example

# Internal logic variables
reaction_time = 1  # 5 s of creation time
danger = z3.And(obstacle_ahead, z3.Or(z3.Not(brake_ok), road_slippery))
time_for_reacting = dist_obstacle / self_speed
should_break = z3.And(danger, 5 * reaction_time < time_for_reacting)

axioms = []

# The symbolic function that determine if the autonomous car should break or not
S_should_break = LogicS(
    logic=Z3Logic(),
    axioms=[],
    inputs_tuple=(obstacle_ahead, brake_ok, road_slippery, self_speed, dist_obstacle),
    query=should_break
)
