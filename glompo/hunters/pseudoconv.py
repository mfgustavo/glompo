
from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger


class PseudoConverged(BaseHunter):

    def __init__(self, threshold: int):
        """ Returns True if the victim's best value has not changed in the last iters iterations """
        self.threshold = threshold

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        i_current = len(log.get_history(victim_opt_id, "fx"))
        i_best_found = log.get_history(victim_opt_id, "i_best")[-1]

        return i_current - i_best_found > self.threshold
