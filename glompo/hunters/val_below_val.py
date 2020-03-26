

from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger

__all__ = ("ValBelowVal",)


class ValBelowVal(BaseHunter):

    def __init__(self):
        """ Returns True if the current best value seen by the hunter falls below the current best value of the
            victim.
        """

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:

        hunt_vals = log.get_history(hunter_opt_id, "fx_best")
        vic_vals = log.get_history(victim_opt_id, "fx_best")
        if all([len(vals) > 0 for vals in [hunt_vals, vic_vals]]):
            return min(hunt_vals) < min(vic_vals)

        return False
