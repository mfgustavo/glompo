
from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger


class ConfidenceWidth(BaseHunter):

    def __init__(self, threshold: float):
        """ Returns True if the standard deviation of the victim's GPR is less than a percentage of the mean.
            The fraction is given by threshold as a value between 0 and 1.
        """
        self.threshold = threshold

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:

        mu, sigma = victim_gpr.estimate_mean()

        return sigma < abs(mu)
