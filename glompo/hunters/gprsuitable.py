
from .basehunter import BaseHunter
from ..core.gpr import GaussianProcessRegression
from ..core.logger import Logger

import numpy as np
import scipy.integrate
import scipy.stats


class GPRSuitable(BaseHunter):

    def __init__(self, tol: float):
        """ Returns True if the means of the GPRs of the hunter and victim arestatistically within tol% of the data
        points used to train the models (tol is a fraction betwwen 0 and 1). Will also return False if the tail of the
        GPR is sharply concave or convex.
        """
        self.tol = tol

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:

        for opt_id, gpr in [(hunter_opt_id, hunter_gpr), (victim_opt_id, victim_gpr)]:
            vals = np.array(log.get_history(opt_id, "fx"))
            mu, sigma = gpr.sample_all(range(len(vals)))
            diffs = np.abs(vals - mu) / vals[0]

            # Check mean matches data
            suitable = all(diffs < self.tol * vals[0])
            if not suitable:
                return False

            # Check tail not too extreme
            x1 = len(vals)
            x2 = x1 + 100
            y1, _ = gpr.sample_all(x1)
            y2, _ = gpr.sample_all(x2)
            m = (y2 - y1) / (x2 - x1)
            if m > 0.02 or m < -0.02:
                return False

        return True
