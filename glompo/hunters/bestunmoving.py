from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger

__all__ = ("BestUnmoving",)


class BestUnmoving(BaseHunter):
    """ Considers the lowest function value seen by the optimizer thus far.
    Returns :obj:`True` if the victim's best value has not changed significantly in a given amount of time.

    Parameters
    ----------
    calls
        Number of function evaluations between comparison points.
    tol
        Tolerance fraction between 0 and 1.

    Returns
    -------
    bool
        :obj:`True` if::

           abs(latest_f_value - f_value_calls_ago) <= abs(f_value_calls_ago * self.tol)
    """

    def __init__(self, calls: int, tol: float = 0):
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
            self.last_result = False
            return self.last_result

        best_at_calls = min(vals[:-self.calls])
        best_at_end = min(vals)
        self.last_result = abs(best_at_end - best_at_calls) <= abs(best_at_calls * self.tol)
        return self.last_result
