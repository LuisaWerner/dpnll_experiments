
import random
from typing import Any


class Unknown:
    """
    Marker object to represent unknown/undefined values.
    Used as the initial value of random variables before they are instantiated.
    Can carry additional optional metadata for explanation purposes.
    """

    def __init__(self, optional=None):
        self.optional = optional

    def __repr__(self):
        return 'unknown'

    def __eq__(self, other):
        return isinstance(other, Unknown)


unknown = Unknown()  # Global singleton for unknown value


class RndVar:
    """
    A discrete random variable with a domain and probability distribution.

    Attributes:
    - name: Identifier
    - domain_distrib: A dictionary mapping values (e.g. True/False) to probabilities
    - value: Current assigned value, or `unknown` if unassigned
    """

    def __init__(self, name, domain_distrib: dict, value: Any = None):
        self.sum_prob = sum(domain_distrib.values())
        assert 0.999 <= self.sum_prob <= 1.001  # Ensure probabilities sum to ~1
        self.name = name
        self.domain_distrib = domain_distrib
        self.value = value if value is not None else unknown

    def defined(self):
        """Returns True if the variable has a concrete (non-unknown) value."""
        return not isinstance(object.__getattribute__(self, "value"), Unknown)

    def domain(self):
        """Returns the set of all possible values this variable can take."""
        return set(self.domain_distrib)

    def sample(self):
        rnd = random.uniform(0.0, self.sum_prob - 0.001)
        sum = 0.0
        for value, prob in self.domain_distrib.items():
            sum += prob
            if sum >= rnd:
                return value

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return f"{repr(self.name)}=={repr(self.value)}"

    def __eq__(self, other):
        return isinstance(other, RndVar) and id(self) == id(other)

    def __copy__(self):
        return RndVar(self.name, {val: prob for val, prob in self.domain_distrib.items()}, self.value)


class BoolRndVar(RndVar):
    """
    A convenient subclass of RndVar for binary (Boolean) variables.
    Uses a Bernoulli distribution with probability p for True.
    """

    def __init__(self, name, p: float, value: Any = None):
        super().__init__(name, {True: p, False: 1.0 - p}, value=value)