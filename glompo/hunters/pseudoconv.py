
from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("PseudoConverged",)


class PseudoConverged(BaseHunter):

    def __init__(self, calls: int, tol: float = 0):
        """ Returns True if the victim's best value has not changed by more than tol fraction in the last calls
            function evaluations where tol is a fraction between 0 and 1.
        """
        super().__init__()
        self.calls = calls
        self.tol = tol

    def __call__(self,
                 log: Logger,
                 regressor: DataRegressor,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        vals = log.get_history(victim_opt_id, "fx_best")
        fcalls = log.get_history(victim_opt_id, "f_call")

        if fcalls[-1] < self.calls:
            return False

        fbest_current = vals[-1]

        i = -1
        nearest_iter = fcalls[-1]
        while fcalls[-1] - nearest_iter < self.calls:
            i += 1
            nearest_iter = fcalls[-2 - i]
        i += 2

        fbest_calls = vals[-i]

        self._last_result = abs(fbest_current - fbest_calls) <= abs(fbest_calls * self.tol)
        return self._last_result
