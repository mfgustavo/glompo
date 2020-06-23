

from typing import *

import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import OptimizerLogger
from ..common.helpers import is_bounds_valid


__all__ = ("ParameterDistance",)


class ParameterDistance(BaseHunter):
    """ Calculates the maximum Euclidean distance in parameter space between the point of lower bounds and the point
        of upper bounds (parameter space length). Calculates the Euclidean distance between the final
        iterations of the hunter and victim (inter-optimizer distance).

        If the fraction between the inter-optimizer distance and parameter space length is less than the provided
        tolerance the optimizers are deemed to be near one another and the condition returns True.

        Note: Use this hunter with caution and only in problems where the parameters have been standardised so that
        every parameter is dimensionless and on the same order of magnitude.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]], relative_distance: float, test_all: bool = False):
        """
        Parameters
        ----------
        bounds: Sequence[Tuple[float, float]]
            Bounds of each parameter.
        relative_distance: float
            The fraction of the maximum distance in the space (from the point at all lower bounds to the point at all
            upper bounds) below which the optimizers are deemed too close and the victim will be killed.
        test_all: bool = False
            If True the distance between victim and all other optimizers is tested, else only the hunter and victim are
            compared.
        """
        super().__init__()
        if isinstance(relative_distance, (float, int)) and relative_distance > 0:
            self.relative_distance = relative_distance
        else:
            raise ValueError("relative_distance should be a positive float.")

        if is_bounds_valid(bounds):
            lower_pt, upper_pt = tuple(np.transpose(bounds))
            self.trans_space_dist = self.distance(lower_pt, upper_pt)

        self.test_all = test_all

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        if self.test_all:
            compare_to = range(1, len(log)+1)
        else:
            compare_to = [hunter_opt_id]

        for opt_id in compare_to:
            if opt_id != victim_opt_id:
                try:
                    h1 = np.array(log.get_history(opt_id, 'x')[-1])
                except IndexError:
                    self.logger.debug(f"Unable to compare to Opt{opt_id}, no points in log")
                    continue
                v1 = np.array(log.get_history(victim_opt_id, 'x')[-1])
                opt_dist = np.sqrt(np.sum((v1 - h1) ** 2))
                ratio = opt_dist / self.trans_space_dist

                self._last_result = ratio <= self.relative_distance
                if self._last_result:
                    self.logger.debug("ParameterDistance: Hunter=%d, Victim=%d, "
                                      "Result=%.2f / %.2f <= %.2f} = "
                                      "%s.",
                                      opt_id,
                                      victim_opt_id,
                                      opt_dist,
                                      self.trans_space_dist,
                                      self.relative_distance,
                                      self._last_result)
                    return self._last_result

        self._last_result = False
        return self._last_result

    @staticmethod
    def distance(pt1: Sequence[float], pt2: Sequence[float]):
        return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2))
