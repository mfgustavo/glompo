

from .basechecker import BaseChecker


class OMaxConvergence(BaseChecker):

    def __init__(self, omax: int):
        """ Convergence is reached after omax optimizers have been started. """
        self.omax = omax

    def converged(self, manager: 'GloMPOManager') -> bool:
        if manager.o_counter >= self.omax:
            return True
        else:
            return False
