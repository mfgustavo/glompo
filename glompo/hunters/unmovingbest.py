import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import OptimizerLogger

__all__ = ("BestUnmoving",)


class BestUnmoving(BaseHunter):

    def __init__(self, calls: int, tol: float = 0):
        """ Returns True if the victim's best value has not changed by more than tol fraction in the last 'calls'
            function evaluations where tol is a fraction between 0 and 1.
        """
        super().__init__()
        self.calls = calls
        self.tol = tol

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        vals = log.get_history(victim_opt_id, "fx_best")
        fcalls = log.get_history(victim_opt_id, "f_call_opt")

        i_crit = np.searchsorted(fcalls - np.max(fcalls) + self.calls, 0) - 1
        if i_crit == -1:
            # If there are insufficient iterations the hunter will return False
            self._last_result = False
            return self._last_result

        self._last_result = abs(vals[-1] - vals[i_crit]) <= abs(vals[i_crit] * self.tol)
        return self._last_result
