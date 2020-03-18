

from .basechecker import BaseChecker
from time import time


__all__ = ("MaxSeconds",)


class MaxSeconds(BaseChecker):

    def __init__(self, tmax: int):
        """ Convergence is reached after omax optimizers have been started. """
        super().__init__()
        self.tmax = tmax

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self._converged = time() - manager.t_start >= self.tmax
        return self._converged
