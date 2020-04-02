

from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("ValBelowAsymptote",)


class ValBelowAsymptote(BaseHunter):

    def __init__(self):
        """ Returns True if the current best value seen by the hunter falls below the 95% confidence threshold of the
            victim.
        """
        self.regressor = DataRegressor()

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, victim_opt_id: int) -> bool:
        hunter_vals = log.get_history(hunter_opt_id, "fx_best")
        victim_y = log.get_history(victim_opt_id, "fx")
        victim_t = range(len(victim_y))

        if len(hunter_vals) > 0:
            mu, sigma = self.regressor.estimate_parameters(victim_t, victim_y,
                                                           parms='asymptote',
                                                           cache_key=victim_opt_id)
            print(f"Optimizer {victim_opt_id} has an asymptote of {mu:.2E} \u00B1 {sigma:.2E}")
            threshold = mu - 2 * sigma

            return hunter_vals[-1] < threshold

        return False
