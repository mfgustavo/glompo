

from .basechecker import BaseChecker


class NOptConverged(BaseChecker):

    def __init__(self, nconv: int):
        """ Convergence is reached after nconv optimizers have been _converged normally. """
        super().__init__()
        self.nconv = nconv

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self._converged = manager.conv_counter >= self.nconv
        return self._converged
