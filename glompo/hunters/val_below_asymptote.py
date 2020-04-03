

from typing import *

import numpy as np

from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor
from ..core.scope import GloMPOScope


__all__ = ("ValBelowAsymptote",)


class ValBelowAsymptote(BaseHunter):

    def __init__(self, scope: Optional[GloMPOScope] = None):
        """ Returns True if the current best value seen by the hunter falls below the 95% confidence threshold of the
            victim.

            Parameters
            ----------
            scope: Optional[GloMPOScope] = None
                Accepts a scope object from the manager. If given the results from the regressions will be used to
                update the asymptotes and uncertainties for each optimizer.
        """
        self.regressor = DataRegressor()
        self.scope = scope

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, victim_opt_id: int) -> bool:
        hunter_vals = log.get_history(hunter_opt_id, "fx_best")
        victim_y = log.get_history(victim_opt_id, "fx")
        victim_t = np.array(range(len(victim_y)))
        print(hunter_vals)
        print(victim_y)
        print(victim_t)

        if len(hunter_vals) > 0:
            mu, sigma = self.regressor.estimate_parameters(victim_t, victim_y,
                                                           parms='asymptote',
                                                           cache_key=victim_opt_id)
            if self.scope:
                self.scope.update_mean(victim_opt_id, mu, sigma)
            threshold = mu - 2 * sigma

            return hunter_vals[-1] < threshold

        return False
