

from .basechecker import BaseChecker


__all__ = ("MaxFuncCalls",)


class MaxFuncCalls(BaseChecker):

    def __init__(self, fmax: int):
        """ Convergence is reached after omax optimizers have been started. """
        super().__init__()
        self.fmax = fmax

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self._converged = manager.f_counter >= self.fmax
        return self._converged
