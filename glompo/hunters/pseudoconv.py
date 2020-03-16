
from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger


class PseudoConverged(BaseHunter):

    def __init__(self, iters: int, tol: float = 0):
        """ Returns True if the victim's best value has not changed by more than tol fraction in the last iters
        iterations where tol is a fraction between 0 and 1. """
        self.iters = iters
        self.tol = tol

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        vals = log.get_history(victim_opt_id, "fx_best")
        if len(vals) < self.iters:
            return False
        else:
            fbest_current = vals[-1]
            fbest_iters = vals[-self.iters]

            return abs(fbest_current - fbest_iters) <= fbest_iters * self.tol
