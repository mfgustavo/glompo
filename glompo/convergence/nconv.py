

from .basechecker import BaseChecker


class NOptConvergence(BaseChecker):

    def __init__(self, nconv: int):
        """ Convergence is reached after nconv optimizers have been converged normally. """
        self.nconv = nconv

    def converged(self, manager: 'GloMPOManager') -> bool:
        if manager.conv_counter >= self.nconv:
            return True
        else:
            return False
