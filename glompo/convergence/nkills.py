from .basechecker import BaseChecker

__all__ = ("MaxKills",)


class MaxKills(BaseChecker):
    """ Returns :obj:`True` after `kills_max` optimizers have been shutdown by the manager. """

    def __init__(self, kills_max: int):
        super().__init__()
        self.kills_max = kills_max

    def __call__(self, manager: 'GloMPOManager') -> bool:
        self.last_result = len(manager.hunt_victims) >= self.kills_max
        return self.last_result
