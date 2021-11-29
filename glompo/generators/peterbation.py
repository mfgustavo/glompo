from typing import Sequence

import numpy as np
from scipy.stats import truncnorm

from .basegenerator import BaseGenerator

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
    scale
        Standard deviation of each parameter. Used here to control how wide the generator should explore around the
        mean.
    """

    def __init__(self, x0: Sequence[float], scale: Sequence[float]):
        super().__init__()
        self.loc = np.array(x0)
        self.scale = np.array(scale)

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        lb = np.array(manager.bounds)[:, 0]
        ub = np.array(manager.bounds)[:, 1]

        a = (lb - self.loc) / self.scale
        b = (ub - self.loc) / self.scale

        x0 = truncnorm.rvs(a, b, self.loc, self.scale)

        return x0
