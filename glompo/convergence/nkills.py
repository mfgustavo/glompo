from .basechecker import BaseChecker

__all__ = ("MaxKills",)


class MaxKills(BaseChecker):

    def __init__(self, kills_max: int):
        """ Convergence is reached after kills_max optimizers have been killed. """
        super().__init__()
        self.kills_max = kills_max

    def __call__(self, manager: 'GloMPOManager') -> bool:
        self._last_result = len(manager.hunt_victims) >= self.kills_max
        return self._last_result
