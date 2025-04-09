from typing import List, Dict, Optional, Union, Set
import copy
import itertools


# ----- Core Term / Clause Classes -----
class Term:
    def __init__(self, name: str, args: Optional[List['Term']] = None):
        self.name = name
        self.args = args or []

    def is_variable(self) -> bool:
        return self.name[0].isupper()

    def __repr__(self):
        if self.args:
            return f"{self.name}({', '.join(map(str, self.args))})"
        return self.name

    def substitute(self, subs: Dict[str, 'Term']) -> 'Term':
        if self.is_variable():
            return subs.get(self.name, self)
        else:
            return Term(self.name, [arg.substitute(subs) for arg in self.args])

    def is_ground(self) -> bool:
        return all(not arg.is_variable() and arg.is_ground() for arg in self.args)


class Fact:
    def __init__(self, head: Term):
        self.head = head

    def __repr__(self):
        return f"{self.head}."


class Rule:
    def __init__(self, head: Term, body: List[Term]):
        self.head = head
        self.body = body

    def __repr__(self):
        return f"{self.head} :- {', '.join(map(str, self.body))}."


Clause = Union[Fact, Rule]


# ----- Unification Logic -----
def unify(x: Term, y: Term, subs: Dict[str, Term]) -> Optional[Dict[str, Term]]:
    if x.is_variable():
        return unify_var(x, y, subs)
    elif y.is_variable():
        return unify_var(y, x, subs)
    elif x.name != y.name or len(x.args) != len(y.args):
        return None
    else:
        for a, b in zip(x.args, y.args):
            subs = unify(a.substitute(subs), b.substitute(subs), subs)
            if subs is None:
                return None
        return subs


def unify_var(var: Term, x: Term, subs: Dict[str, Term]) -> Optional[Dict[str, Term]]:
    if var.name in subs:
        return unify(subs[var.name], x, subs)
    elif x.is_variable() and x.name in subs:
        return unify(var, subs[x.name], subs)
    else:
        subs[var.name] = x
        return subs


# ----- Debug Print -----
DEBUG = False


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# ----- Solver Class -----
class PrologSolver:
    def __init__(self, clauses: List[Clause]):
        self.clauses = clauses
        self._rename_id = itertools.count()
        self.index = self._build_index(clauses)
        self.ground_facts = self._build_ground_facts(clauses)
        self.memo_failed: set[str] = set()

    def _build_ground_facts(self, clauses: List[Clause]) -> dict[str, tuple[dict[str, int], list]]:
        ground_facts = {}
        for idx, clause in enumerate(clauses):
            if isinstance(clause, Fact) and clause.head.is_ground():
                key = self._predicate_key(clause.head)
                tmp1, tmp2 = ground_facts.get(key, ({}, []))
                tmp1[repr(clause.head)] = idx
                tmp2.append((idx, clause))
                ground_facts[key] = tmp1, tmp2
        return ground_facts

    def _build_index(self, clauses: List[Clause]):
        index = {}
        for i, clause in enumerate(clauses):
            if not(isinstance(clause, Fact) and clause.head.is_ground()):
                key = self._predicate_key(clause.head)
                index.setdefault(key, []).append((i, clause))
        return index

    def _predicate_key(self, term: Term) -> str:
        return f"{term.name}/{len(term.args)}"

    def solve(
            self,
            goals: List[Term],
            subs: Dict[str, Term] = None,
            used_indices: Set[int] = None,
            visited_goals: List[str] = None,
            depth: int = 0
    ) -> Optional[Set[int]]:
        if subs is None:
            subs = {}
        if used_indices is None:
            used_indices = set()
        if visited_goals is None:
            visited_goals = []

        indent = "  " * depth
        if not goals:
            debug_print(f"{indent}✔️ All goals satisfied")
            return used_indices

        current_goal = goals[0].substitute(subs)
        rest_goals = goals[1:]

        goal_key = repr(current_goal)
        if current_goal.is_ground():
            if goal_key in self.memo_failed:
                debug_print(f"{indent}MEMOIZATION HIT Memo hit (fail): {current_goal}")
                return None
        debug_print(f"{indent}🔍 Solving: {current_goal}")

        if goal_key in visited_goals:
            debug_print(f"{indent}🔁 Loop detected, skipping: {current_goal}")
            return None

        key = self._predicate_key(current_goal)

        # Shortcut for ground facts
        if current_goal.is_ground() and goal_key in self.ground_facts.get(key, (set(), []))[0]:
            debug_print(f"{indent} Ground fact matched from cache: {current_goal}")
            return used_indices | {self.ground_facts[key][0][goal_key]}

        if current_goal.is_ground():
            checklist = self.index.get(key, [])
        else:
            checklist = self.ground_facts.get(key, (set(), []))[1] + self.index.get(key, [])

        for idx, clause in checklist:
            if clause.head.name != current_goal.name or len(clause.head.args) != len(current_goal.args):
                continue

            renamed_clause = self._rename_clause(clause)
            head = renamed_clause.head
            new_subs = unify(current_goal, head, copy.deepcopy(subs))

            if new_subs is not None:
                debug_print(f"{indent}✅ Clause #{idx} matches: {clause}")
                new_used = used_indices | {idx}

                if isinstance(renamed_clause, Fact):
                    result = self.solve(
                        rest_goals, new_subs, new_used,
                        visited_goals + [goal_key], depth + 1
                    )
                    if result is not None:
                        return result

                elif isinstance(renamed_clause, Rule):
                    new_goals = renamed_clause.body + rest_goals
                    result = self.solve(
                        new_goals, new_subs, new_used,
                        visited_goals + [goal_key], depth + 1
                    )
                    if result is not None:
                        return result

        debug_print(f"{indent}❌ No clause matches for: {current_goal}")
        if current_goal.is_ground():
            self.memo_failed.add(goal_key)
            debug_print(f"{indent}❌ Memoizing failure: {current_goal}")
        return None

    def _rename_clause(self, clause: Clause) -> Clause:
        counter = next(self._rename_id)
        var_map = {}

        def rename(term: Term) -> Term:
            if term.is_variable():
                if term.name not in var_map:
                    var_map[term.name] = Term(f"{term.name}_{counter}")
                return var_map[term.name]
            return Term(term.name, [rename(arg) for arg in term.args])

        head = rename(clause.head)
        if isinstance(clause, Fact):
            return Fact(head)
        else:
            body = [rename(term) for term in clause.body]
            return Rule(head, body)


# ----- Utility Entry Point -----
def solve_clauses(clauses: List[Clause], query: List[Term]) -> Optional[Set[int]]:
    solver = PrologSolver(clauses)
    return solver.solve(query)


# --- Testing the Solver ---

def test_reachability_with_clause_indices():
    clauses = [
        Fact(Term("edge", [Term("a"), Term("b")])),
        Fact(Term("edge", [Term("b"), Term("c")])),
        Fact(Term("edge", [Term("c"), Term("d")])),
        Fact(Term("edge", [Term("d"), Term("e")])),
        Fact(Term("edge", [Term("e"), Term("a")])),  # cycle
        Rule(Term("reachable", [Term("X"), Term("Y")]),
             [Term("edge", [Term("X"), Term("Y")])]),
        Rule(Term("reachable", [Term("X"), Term("Y")]),
             [Term("edge", [Term("X"), Term("Z")]), Term("reachable", [Term("Z"), Term("Y")])])
    ]

    query = [Term("reachable", [Term("a"), Term("e")])]

    result = solve_clauses(clauses, query)
    print("\nUsed clause indices:", result)


def test_unsatisfiable_query():
    clauses = [
        Fact(Term("edge", [Term("a"), Term("b")])),
        Fact(Term("edge", [Term("b"), Term("c")])),
        Rule(Term("reachable", [Term("X"), Term("Y")]),
             [Term("edge", [Term("X"), Term("Y")])]),
        Rule(Term("reachable", [Term("X"), Term("Y")]),
             [Term("edge", [Term("X"), Term("Z")]), Term("reachable", [Term("Z"), Term("Y")])])
    ]

    query = [Term("reachable", [Term("a"), Term("z")])]  # 'z' is not reachable

    result = solve_clauses(clauses, query)
    if result is None:
        print("❌ No proof exists for the query.")
    else:
        print("✅ Proof found using clause indices:", result)

