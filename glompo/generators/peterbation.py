

from typing import *

import numpy as np
from scipy.stats import truncnorm

from .basegenerator import BaseGenerator


__all__ = ("PerterbationGenerator",)


class PerterbationGenerator(BaseGenerator):
    """ Randomly generates parameter vectors by drawing from truncated normally distributed numbers centered around a
        provided vector and bound by given bounds. Good for reparameterisation efforts where a good candidate is already
        available, however, this may drastically limit the exploratory nature of GloMPO.
    """

    def __init__(self, x0: Sequence[float], bounds: Sequence[Tuple[float, float]], scale: Sequence[float]):
        """

        Parameters
        ----------
        x0 : Sequence[float]
            Center point for each parameter
        bounds : Sequence[Tuple[float, float]]
            Min and max bounds for each parameter
        scale : Sequence[float]
            Standard deviation of each parameter. Used here to control how wide the generator should explore around the
            mean.
        """
        self.n_params = len(x0)
        self.loc = np.array(x0)

        if len(bounds) != self.n_params:
            raise ValueError("Bounds and x0 not the same length")

        for i, bnd in enumerate(bounds):
            if bnd[0] >= bnd[1]:
                raise ValueError("Invalid bounds encountered. Min and max bounds may not be equal nor may they be in"
                                 "the opposite order. ")
            if not np.all(np.isfinite(bnd)):
                raise ValueError("Non-finite bounds found.")
            if self.loc[i] <= bnd[0] or self.loc[i] >= bnd[1]:
                raise ValueError("Value in x0 out of bounds.")

        bnds = np.array(bounds)
        self.min = bnds[:, 0]
        self.max = bnds[:, 1]

        self.scale = np.array(scale)

    def generate(self) -> np.ndarray:
        call = []
        for i in range(self.n_params):
            a = (self.min[i] - self.loc[i]) / self.scale[i]
            b = (self.max[i] - self.loc[i]) / self.scale[i]
            mu = self.loc[i]
            sigma = self.scale[i]
            call.append(truncnorm.rvs(a, b, mu, sigma))
        return np.array(call)
