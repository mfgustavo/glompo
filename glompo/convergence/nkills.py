

from .basechecker import BaseChecker


class KillsMaxConvergence(BaseChecker):

    def __init__(self, kills_max: int):
        """ Convergence is reached after kills_max optimizers have been killed. """
        self.kills_max = kills_max

    def converged(self, manager: 'GloMPOManager') -> bool:
        if manager.kill_counter >= self.kills_max:
            return True
        else:
            return False
