
from .basehunter import BaseHunter
from ..core.logger import Logger


__all__ = ("MinVictimTrainingPoints",)


class MinVictimTrainingPoints(BaseHunter):

    def __init__(self, min_pts: int):
        """ Returns True if the victim has more than min_pts in its GPR. """
        if min_pts > 0 and isinstance(min_pts, int):
            self.min_pts = min_pts
        else:
            raise ValueError("min_pts must be a positive integer.")

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, victim_opt_id: int) -> bool:
        return len(victim_gpr.training_coords()) >= self.min_pts
