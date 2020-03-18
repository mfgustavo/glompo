

from .basechecker import BaseChecker


__all__ = ("MaxOptsStarted",)


class MaxOptsStarted(BaseChecker):

    def __init__(self, omax: int):
        """ Convergence is reached after omax optimizers have been started. """
        super().__init__()
        self.omax = omax

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self._converged = manager.o_counter >= self.omax
        return self._converged
