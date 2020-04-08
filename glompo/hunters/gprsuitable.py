
from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("GPRSuitable",)


class GPRSuitable(BaseHunter):

    def __init__(self, tol: float):
        """ Returns True if the means of the GPRs of the hunter and victim are statistically within tol% of the data
            points used to train the models (tol is a fraction betwwen 0 and 1). Will also return False if the tail of
            the GPR is sharply concave or convex.
        """
        self.tol = tol

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        return hunter_gpr.is_suitable(self.tol) and victim_gpr.is_suitable(self.tol)
