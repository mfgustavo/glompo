import random

from .basehunter import BaseHunter
from ..core.optimizerlogger import OptimizerLogger

__all__ = ("TimeAnnealing",)


class TimeAnnealing(BaseHunter):

    def __init__(self, crit_ratio: float = 1):
        """ This condition calculates the quotient (N_hunter_fcalls / N_victim_fcalls). A random number is
            then generated between zero and crit_ratio. Only if the quotient is larger than this number does the victim
            remain alive.

            Parameters
            ----------
            crit_ratio: float = 1
                Critical ratio of function calls (not necessarily iterations) between hunter and victim above which the
                victim is guaranteed to survive. Values lower than one are looser and allow the victim to survive even
                if it has been in operation longer than the hunter. Values higher than one are stricter and may kill
                the victim even if it has iterated fewer times than the hunter.
        """
        super().__init__()
        if isinstance(crit_ratio, (float, int)) and crit_ratio > 0:
            self.crit_ratio = crit_ratio
        else:
            raise ValueError("threshold should be a positive float.")

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        n_hunter = log.len(hunter_opt_id)
        n_victim = log.len(victim_opt_id)

        ratio = n_hunter / n_victim
        test_num = random.uniform(0, self.crit_ratio)

        self._last_result = test_num > ratio
        return self._last_result
