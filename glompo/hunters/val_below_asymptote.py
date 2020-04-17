

import numpy as np

from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("ValBelowAsymptote",)


class ValBelowAsymptote(BaseHunter):

    """ Performs a non-linear Bayesian regression on the victim's iteration history. Returns True if the hunter has
        calculated values below the confidence interval projection of the victim. The significance of this confidence
        interval is set by during initialisation.
    """

    def __init__(self, significance: int = 0.95, nwalkers: int = 20, nsteps: int = 5000):
        """ Parameters
            ----------
            significance: int = 0.95
                Confidence value of the interval returned by the Bayesian regression. Must be in the interval (0, 1)
                exclusive.
            nwalkers: int
                Number of MCMC chains run in parallel by the sampler.
            nsteps: int
                Number of samples taken by each walker.
        """
        super().__init__()
        self.significance = significance
        self.nwalkers = nwalkers
        self.nsteps = nsteps

    def __call__(self,
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
                                                  nsteps=self.nsteps,
                                                  nwalkers=self.nwalkers,
                                                  significance=self.significance,
                                                  cache_key=victim_opt_id)

            if len(result) == 3:
                _, low, _ = tuple(victim_y[-1] * val for val in result)
                self._last_result = hunter_vals[-1] < low
                return self._last_result

        self._last_result = False
        return self._last_result
