from .basehunter import BaseHunter
from ..core.optimizerlogger import OptimizerLogger

__all__ = ("MinIterations",)


class MinIterations(BaseHunter):

    def __init__(self, min_pts: int):
        """ Returns True if the victim has iterated at least min_pts times. Use cautiously in conjunction with
            multiple optimizer types since iterations are often defined very differently and do not necessarily
            equal one function call.
        """
        super().__init__()
        self.min_pts = min_pts

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        items = log.get_history(victim_opt_id, 'iter_id')[-1]

        self._last_result = items >= self.min_pts
        return self._last_result
