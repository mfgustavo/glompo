
from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger

import numpy as np
import scipy.integrate
import scipy.stats


class GPRSuitable(BaseHunter):

    def __init__(self):
        """ Returns True if the GPRs of the hunter and victim adequately fit the data and have not significantly
        diverged. Implements a Bayes t-test to determine if the GPR mean is sufficiently close to the data points
        within it.
        """
        pass

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:

        for opt_id in [hunter_opt_id, victim_opt_id]:
            vals = np.array(log.get_history(opt_id, "fx"))
            mu, sigma = hunter_gpr.sample_all(range(len(vals)), scaled=True)
            diffs = np.abs(vals - mu)
            suitable = self.bayes_factor(diffs, 0) > 1
            if not suitable:
                return False

        return True

    @staticmethod
    def bayes_factor(samples: np.ndarray, h0: float, r_factor=1.0) -> float:
        """ Calculates the Bayes Factor for the mean of a given set of samples
            against a null hypothesis. Presented as the likelihood of the null
            against the alternative.

            Parameters
            ----------
            samples : ndarray
                An array of samples that will be tested.
            h0 : float
                The value of the null hypothesis being tested.
            r_factor : float
                Effect size scale factor, default is one.

            Notes
            -----
            Follows the two-sided construction of (Rouder et al., 2009)
        """
        # Calculate t statistic and degrees of freedom
        num_samples = len(samples)
        deg_free = num_samples - 1
        t_stat, _ = scipy.stats.ttest_1samp(samples, h0)

        # Calculate Bayes Factor
        def integrand(g):
            return ((1 + num_samples * g * r_factor ** 2) ** -0.5) * \
                   ((1 + ((t_stat ** 2) / (deg_free * (1 + num_samples * g * r_factor ** 2)))) ** (- num_samples / 2))\
                   * ((2 * np.pi) ** -0.5) * (g ** -1.5) * np.exp(- 0.5 / g)

        integral = scipy.integrate.quad(integrand, 0, np.inf)[0]

        bf_01 = ((1 + (t_stat ** 2) / deg_free) ** (- num_samples / 2)) / integral

        return bf_01
