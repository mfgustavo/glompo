

from time import time
from .basechecker import BaseChecker


__all__ = ("MaxSeconds",)


class MaxSeconds(BaseChecker):

    def __init__(self, tmax: int):
        """ Convergence is reached after tmax seconds have elapsed. """
        super().__init__()
        self.tmax = tmax

    def __call__(self, manager: 'GloMPOManager') -> bool:
        self._last_result = time() - manager.t_start >= self.tmax
        return self._last_result
