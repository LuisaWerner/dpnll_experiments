import abc

from dpnl.core.symbolic import Symbolic, Input


class CachingHash(abc.ABC):
    """
    Abstract base class representing a caching hash. A caching hash is a function H such that for all I1 and I2 input
    of the symbolic function S, if H(I1) = H(I2) then the set of valuation of the undefined variables in I1 is equal
    to the set of valuation of the undefined variables in I2. This allows to do caching in the DPNL algorithm.
    """
    def __init__(self, S: Symbolic):
        self.S = S

    @abc.abstractmethod
    def __call__(self, I: Input):
        pass
