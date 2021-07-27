from typing import Sequence, Tuple

import numpy as np

from .basehunter import BaseHunter
from ..common.helpers import distance, is_bounds_valid
from ..core.optimizerlogger import BaseLogger

__all__ = ("ParameterDistance",)


class ParameterDistance(BaseHunter):
    """ Terminates optimizers which are too close in parameter space.

    Parameters
    ----------
    bounds
        Bounds of each parameter.
    relative_distance
        Fraction (between 0 and 1) of the maximum distance in the space (from the point at all lower bounds to the point
        at all upper bounds) below which the optimizers are deemed too close and the victim will be killed.
    test_all
        If :obj:`True` the distance between victim and all other optimizers is tested, else only the hunter and victim
        are compared.

    Returns
    -------
    bool
        Returns :obj:`True` if optimizers are calculated to be too close together.

    Notes
    -----
    Calculates the maximum Euclidean distance in parameter space between the point of lower bounds and the point of
    upper bounds (:code:`max_distance`). Calculates the Euclidean distance between the final iterations of the hunter
    and victim (:code:`inter_optimizer_distance`). Returns :obj:`True` if::

        inter_optimizer_distance <= max_distance * relative_distance

    .. caution::

        Use this hunter with care and only in problems where the parameters have been standardised so that every
        parameter is dimensionless and on the same order of magnitude.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]], relative_distance: float, test_all: bool = False):
        super().__init__()
        if isinstance(relative_distance, (float, int)) and relative_distance > 0:
            self.relative_distance = relative_distance
        else:
            raise ValueError("relative_distance should be a positive float.")

        if is_bounds_valid(bounds):
            lower_pt, upper_pt = tuple(np.transpose(bounds))
            self.trans_space_dist = distance(lower_pt, upper_pt)

        self.test_all = test_all

    def __call__(self,
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        if self.test_all:
            compare_to = range(1, log.n_optimizers + 1)
        else:
            compare_to = [hunter_opt_id]

        for opt_id in compare_to:
            if opt_id != victim_opt_id:
                try:
                    h1 = np.array(log.get_history(opt_id, 'x')[-1])
                except IndexError:
                    self.logger.debug("Unable to compare to Opt%d, no points in log", opt_id)
                    continue
                v1 = np.array(log.get_history(victim_opt_id, 'x')[-1])
                opt_dist = distance(h1, v1)
                ratio = opt_dist / self.trans_space_dist

                self.last_result = ratio <= self.relative_distance
                if self.last_result:
                    self.logger.debug("ParameterDistance: Hunter=%d, Victim=%d, "
                                      "Result=%.2f / %.2f <= %.2f} = "
                                      "%s.",
                                      opt_id,
                                      victim_opt_id,
                                      opt_dist,
                                      self.trans_space_dist,
                                      self.relative_distance,
                                      self.last_result)
                    return self.last_result

        self.last_result = False
        return self.last_result
