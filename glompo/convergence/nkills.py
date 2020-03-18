

from .basechecker import BaseChecker


__all__ = ("MaxKills",)


class MaxKills(BaseChecker):

    def __init__(self, kills_max: int):
        """ Convergence is reached after kills_max optimizers have been killed. """
        super().__init__()
        self.kills_max = kills_max

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self._converged = manager.kill_counter >= self.kills_max
        return self._converged
