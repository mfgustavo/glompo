from typing import Sequence, Tuple

import numpy as np
from scipy.stats import truncnorm

from .basegenerator import BaseGenerator
from ..common.helpers import is_bounds_valid

__all__ = ("PerturbationGenerator",)


class PerturbationGenerator(BaseGenerator):
    """ Randomly generates parameter vectors near a given point.
    Draws samples from a truncated multivariate normal distributed centered around a provided vector and bound by given
    bounds. Good for parametrisation efforts where a good candidate is already available, however, this may drastically
    limit the exploratory nature of GloMPO.

    Parameters
    ----------
    x0
        Center point for each parameter
    bounds
        Min and max bounds for each parameter
    scale
        Standard deviation of each parameter. Used here to control how wide the generator should explore around the
        mean.
    """

    def __init__(self, x0: Sequence[float], bounds: Sequence[Tuple[float, float]], scale: Sequence[float]):
        super().__init__()
        self.n_params = len(x0)
        self.loc = np.array(x0)

        if len(bounds) != self.n_params:
            raise ValueError("Bounds and x0 not the same length")

        if is_bounds_valid(bounds):
            bnds = np.array(bounds)
            self.min = bnds[:, 0]
            self.max = bnds[:, 1]

        self.scale = np.array(scale)

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        call = []
        for i in range(self.n_params):
            a = (self.min[i] - self.loc[i]) / self.scale[i]
            b = (self.max[i] - self.loc[i]) / self.scale[i]
            mu = self.loc[i]
            sigma = self.scale[i]
            call.append(truncnorm.rvs(a, b, mu, sigma))
        return np.array(call)
