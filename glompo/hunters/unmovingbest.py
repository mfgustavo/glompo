from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger

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
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        vals = log.get_history(victim_opt_id, "fx")
        fcalls = log.len(victim_opt_id)

        if fcalls <= self.calls:
            # If there are insufficient iterations the hunter will return False
            self._last_result = False
            return self._last_result

        best_at_calls = min(vals[:-self.calls])
        best_at_end = min(vals)
        self._last_result = abs(best_at_end - best_at_calls) <= abs(best_at_calls * self.tol)
        return self._last_result
