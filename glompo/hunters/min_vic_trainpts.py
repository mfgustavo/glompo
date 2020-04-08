
from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("MinVictimTrainingPoints",)


class MinVictimTrainingPoints(BaseHunter):

    def __init__(self, min_pts: int):
        """ Returns True if the victim has more than min_pts in its GPR. """
        if min_pts > 0 and isinstance(min_pts, int):
            self.min_pts = min_pts
        else:
            raise ValueError("min_pts must be a positive integer.")

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        items = len(log.get_history(victim_opt_id))
        return items >= self.min_pts
