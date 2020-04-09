

import numpy as np

from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("ValBelowAsymptote",)


class ValBelowAsymptote(BaseHunter):

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        hunter_vals = log.get_history(hunter_opt_id, "fx_best")
        victim_y = log.get_history(victim_opt_id, "fx")
        victim_t = np.array(range(len(victim_y)))

        if len(hunter_vals) > 0:
            result = regressor.estimate_posterior(victim_t, victim_y,
                                                  parms='asymptote',
                                                  nsteps=3000,
                                                  nwalkers=10,
                                                  cache_key=victim_opt_id)

            if len(result) == 3:
                med, low, upp = tuple(victim_y[-1] * val for val in result)
                return hunter_vals[-1] < low

        return False
