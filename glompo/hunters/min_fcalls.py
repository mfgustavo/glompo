from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger

__all__ = ("MinFuncCalls",)


class MinFuncCalls(BaseHunter):
    """ Keeps an optimizer alive for a minimum number of function evaluations.

    Parameters
    ----------
    min_pts
        Minimum number of points for which an optimizer should be kept alive.

    Returns
    -------
    bool
        Returns :obj:`True` after the function has been evaluated at least `min_pts` times.
    """

    def __init__(self, min_pts: int):
        super().__init__()
        if min_pts > 0 and isinstance(min_pts, int):
            self.min_pts = min_pts
        else:
            raise ValueError("min_pts must be a positive integer.")

    def __call__(self,
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        self.last_result = log.len(victim_opt_id) >= self.min_pts
        return self.last_result
