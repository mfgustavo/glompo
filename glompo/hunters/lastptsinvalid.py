

import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import OptimizerLogger


__all__ = ("LastPointsInvalid",)


class LastPointsInvalid(BaseHunter):

    def __init__(self, n_iters: int = 1):
        """ Some pathological functions may have undefined regions within them or combinations of parameters which
            return non-finite results. This hunter can be used to terminate an optimizer which has not found a valid
            iteration within its last n_iters number of iterations.
        """
        super().__init__()
        self.n_iters = n_iters

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:

        fcalls = log.get_history(victim_opt_id, "fx")[-self.n_iters:]
        self._last_result = len(fcalls) >= self.n_iters and np.isinf(fcalls).all()
        return self._last_result
