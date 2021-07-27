import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger

__all__ = ("LastPointsInvalid",)


class LastPointsInvalid(BaseHunter):
    """ Checks for non-numerical solutions.
    Some pathological functions may have undefined regions within them or combinations of parameters which return
    non-finite results.

    Parameters
    ----------
    n_iters
        Number of allowed invalid function evaluations.

    Returns
    -------
    bool
        Returns :obj:`True` if the optimizer fails to find a valid function evaluation in the last `n_iters` function
        evaluations.
    """

    def __init__(self, n_iters: int = 1):
        super().__init__()
        self.n_iters = n_iters

    def __call__(self,
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        fcalls = log.get_history(victim_opt_id, "fx")[-self.n_iters:]
        self.last_result = len(fcalls) >= self.n_iters and not np.isfinite(fcalls).any()
        return self.last_result
