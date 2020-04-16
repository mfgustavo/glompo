

import random

from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("TimeAnnealing",)


class TimeAnnealing(BaseHunter):

    def __init__(self, crit_ratio: float = 1):
        """ This condition calculates the quotient (N_hunter_iterations / N_victim_iterations). A random number is
            then generated between zero and crit_ratio. Only if the quotient is larger than this number does the victim
            remain alive.

            Parameters
            ----------
            crit_ratio: float = 1
                Critical ratio of iterations between hunter and victim above which the victim is guarenteed to survive.
                Values lower than one are looser and allow the victim to survive even if it has been in operation
                longer than the hunter. Values higher than one are stricter and may kill the victim even if it has
                iterated fewer times than the hunter.
        """
        if isinstance(crit_ratio, (float, int)) and crit_ratio > 0:
            self.crit_ratio = crit_ratio
        else:
            raise ValueError("threshold should be a positive float.")

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        n_hunter = len(log.get_history(hunter_opt_id, "fx"))
        n_victim = len(log.get_history(victim_opt_id, "fx"))

        ratio = n_hunter / n_victim
        test_num = random.uniform(0, self.crit_ratio)

        print(f"Ratio is {ratio:.2f}. Test is {test_num:.2f}. Kill is {test_num > ratio}")

        return test_num > ratio
