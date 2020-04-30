

from typing import *

import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import OptimizerLogger
from ..core.regression import DataRegressor
from ..common.helpers import is_bounds_valid


__all__ = ("ParameterDistance",)


class ParameterDistance(BaseHunter):
    """ Calculates the maximum Euclidean distance in parameter space between the point of lower bounds and the point
        of upper bounds (parameter space length). Calculates the Euclidean distance between the final
        iterations of the hunter and victim (inter-optimizer distance).

        If the fraction between the inter-optimizer distance and parameter space length is less than the provided
        tolerance the optimizers are deemed to be near one another and the hunter returns True.

        Note: Use this hunter with caution and only in problems where the parameters have been standardised so that
        every parameter is dimensionless and on the same order of magnitude.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]], relative_distance: float):
        super().__init__()
        if isinstance(relative_distance, (float, int)) and relative_distance > 0:
            self.relative_distance = relative_distance
        else:
            raise ValueError("relative_distance should be a positive float.")

        if is_bounds_valid(bounds):
            lower_pt, upper_pt = tuple(np.transpose(bounds))
            self.trans_space_dist = self.distance(lower_pt, upper_pt)

    def __call__(self,
                 log: OptimizerLogger,
                 regressor: DataRegressor,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        h1 = np.array(log.get_history(hunter_opt_id, 'x')[-1])
        v1 = np.array(log.get_history(victim_opt_id, 'x')[-1])

        opt_dist = np.sqrt(np.sum((v1 - h1) ** 2))
        ratio = opt_dist / self.trans_space_dist

        self._last_result = ratio <= self.relative_distance
        self.logger.debug("ParameterDistance: Hunter=%d, Victim=%d, "
                          "Result=%.2f / %.2f <= %.2f} = "
                          "%s.",
                          hunter_opt_id,
                          victim_opt_id,
                          opt_dist,
                          self.trans_space_dist,
                          self.relative_distance,
                          self._last_result)
        return self._last_result

    @staticmethod
    def distance(pt1: Sequence[float], pt2: Sequence[float]):
        return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2))
