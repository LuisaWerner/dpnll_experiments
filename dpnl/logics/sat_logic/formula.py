# Logic objects

class Literal:
    def __init__(self, name, idx: int, idx2lit: dict = None):
        self.name = name
        self.idx = idx
        self.idx2lit = idx2lit
        if idx2lit is not None:
            idx2lit[idx] = self

    def __neg__(self):
        return Literal(self.name, -self.idx, self.idx2lit)

    def __repr__(self):
        if self.idx < 0:
            return f"-{self.name}"
        else:
            return repr(self.name)

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return isinstance(other, Literal) and self.idx == other.idx

    def get_literals(self):
        return [self.idx]


class Clause:
    def __init__(self, literals: list[Literal]):
        self.literals = literals

    def get_literals(self):
        return [lit.idx for lit in self.literals]

    def __repr__(self):
        return f"Or({repr(self.literals)})"


class Conj:
    def __init__(self, literals: list[Literal]):
        self.literals = literals

    def __repr__(self):
        return f"And({repr(self.literals)})"