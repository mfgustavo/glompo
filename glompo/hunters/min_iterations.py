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
        if min_pts > 0 and isinstance(min_pts, int):
            self.min_pts = min_pts
        else:
            raise ValueError("min_pts must be a positive integer.")

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        items = len(log.get_history(victim_opt_id))

        self._last_result = items >= self.min_pts
        return self._last_result
