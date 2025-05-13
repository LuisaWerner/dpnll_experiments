import abc
import random
from copy import deepcopy

from typing import Any
import cloudpickle

from dpnl.core.variable import RndVar


class Input:
    """
    Abstract base class representing a probabilistic input. This abstraction allow to manage complex structures of
    inputs with RndVar inside easily.
    """
    def __init__(self, probabilistic_attributes: set[str]):
        self.probabilistic_attributes = probabilistic_attributes

    def __iter__(self):
        for prob_attr in self.probabilistic_attributes:
            yield from self._collect_rnd_vars(getattr(self, prob_attr))

    def _collect_rnd_vars(self, container: Any):
        if isinstance(container, RndVar):
            yield container
        else:
            try:
                for sub_container in iter(container):
                    yield from self._collect_rnd_vars(sub_container)
            except TypeError:
                pass  # Ignore non-iterables

    def randomize_probabilities(self):
        for var in self:
            for val in var.domain_distrib:
                var.domain_distrib[val] = float(random.randint(1, 1000))
            total = sum(var.domain_distrib.values())
            for val in var.domain_distrib:
                var.domain_distrib[val] /= total


class Symbolic(abc.ABC):
    """
    Abstract base class representing a symbolic function S. It takes as input I of type Input where all RndVar have been
    defined and returns its corresponding output.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, I: Input):
        pass
