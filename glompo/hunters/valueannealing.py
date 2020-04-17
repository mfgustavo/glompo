
import random

from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("ValueAnnealing",)


class ValueAnnealing(BaseHunter):
    """ This condition calculates the quotient (fbest_hunter / fbest_victim). A random number is
        then generated between zero and one. Only if the quotient is larger than this number does the victim
        remain alive.
    """

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        f_hunter = log.get_history(hunter_opt_id, "fx_best")[-1]
        f_victim = log.get_history(victim_opt_id, "fx_best")[-1]

        ratio = f_hunter / f_victim
        test_num = random.uniform(0, 1)

        self._kill_result = test_num > ratio
        return self._kill_result
