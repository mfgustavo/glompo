
from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger


class MinVictimTrainingPoints(BaseHunter):

    def __init__(self, min_pts: int):
        """ Returns True if the victim has more than min_pts in its GPR. """
        self.min_pts = min_pts

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        return len(victim_gpr.training_coords()) > self.min_pts
