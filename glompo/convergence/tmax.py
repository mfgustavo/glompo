from time import time
from typing import Optional

from .basechecker import BaseChecker

__all__ = ("MaxSeconds",)


class MaxSeconds(BaseChecker):
    """ Time based convergence criteria.
    Differentiates between session time and overall optimisation time, the difference is only relevant in the case where
    checkpointing is being used.

    Parameters
    ----------
    session_max
        Maximum time in seconds which may elapse from the time GloMPOManager.start_manager() is called.
    overall_max
        Maximum time in seconds which may elapse in total over all optimisation sessions. In other words the
        total time used previously is loaded from the checkpoint. In the case where checkpoint is not used, both
        quantities are equivalent.

    Notes
    -----
    For the avoidance of doubt, both times cannot be used simultaneously. If required rather initialise two
    instances with one condition each.
    """

    def __init__(self, *, session_max: Optional[float] = None, overall_max: Optional[float] = None):
        assert bool(session_max) ^ bool(overall_max)
        super().__init__()
        self.session_max = session_max
        self.overall_max = overall_max

    def __call__(self, manager: 'GloMPOManager') -> bool:
        t_total = time() - manager.t_start if manager.t_start else 0
        cond = self.session_max

        if self.overall_max:
            t_total += manager.t_used
            cond = self.overall_max

        self.last_result = t_total >= cond
        return self.last_result
