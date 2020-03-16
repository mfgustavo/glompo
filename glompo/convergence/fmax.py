

from .basechecker import BaseChecker


class MaxFuncCalls(BaseChecker):

    def __init__(self, fmax: int):
        """ Convergence is reached after omax optimizers have been started. """
        super().__init__()
        self.fmax = fmax

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self.converged = manager.f_counter >= self.fmax
        return self.converged

    def __str__(self):
        return f"{self.__class__.__name__}()"
