
from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger

import numpy as np


__all__ = ("GPRSuitable",)


class GPRSuitable(BaseHunter):

    def __init__(self, tol: float):
        """ Returns True if the means of the GPRs of the hunter and victim arestatistically within tol% of the data
        points used to train the models (tol is a fraction betwwen 0 and 1). Will also return False if the tail of the
        GPR is sharply concave or convex.
        """
        self.tol = tol

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        # print(f"hunter mean suitable {bool(hunter_gpr.is_mean_suitable(self.tol))}")
        # print(f"hunter tail suitable {bool(hunter_gpr.is_tail_suitable())}")
        # print(f"victim mean suitable {bool(victim_gpr.is_mean_suitable(self.tol))}")
        # print(f"victim tail suitable {bool(victim_gpr.is_tail_suitable())}")
        return hunter_gpr.is_suitable(self.tol) and victim_gpr.is_suitable(self.tol)
