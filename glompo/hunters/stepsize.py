from typing import Sequence, Tuple

import numpy as np

from .basehunter import BaseHunter
from ..common.helpers import distance, is_bounds_valid
from ..core.optimizerlogger import OptimizerLogger

__all__ = ("StepSize",)


class StepSize(BaseHunter):

    def __init__(self, bounds: Sequence[Tuple[float, float]], calls: int, relative_tol: float = 0.05):
        """ Returns True if the victim's average step size over the last calls function evaluations is less than
            relative_tol * maximum_parameter_space_distance. In other words this hunter will kill an optimizer that is
            excessively focused on one area of parameter space.
        """
        super().__init__()
        self.calls = calls
        self.tol = relative_tol

        if is_bounds_valid(bounds):
            lower_pt, upper_pt = tuple(np.transpose(bounds))
            self.trans_space_dist = distance(lower_pt, upper_pt)

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        trials = log.get_history(victim_opt_id, "x")[::-1]
        fcalls = log.get_history(victim_opt_id, "f_call_opt")[::-1]

        dists = []
        if fcalls[0] > self.calls:
            for i, fcall in enumerate(fcalls[1:], 1):
                if fcalls[0] - fcall < self.calls:
                    dists.append(distance(trials[0], trials[i]))
                else:
                    break

        if len(dists) > 0:
            mean_dist = np.mean(dists)
            self._last_result = mean_dist / self.trans_space_dist <= self.tol
            self.logger.debug(f"Distances: {dists}\n"
                              f"Mean: {mean_dist}\n"
                              f"Maximum Trans Space Distance: {self.trans_space_dist}\n"
                              f"Returning: {self._last_result}")
        else:
            self._last_result = False

        return self._last_result
