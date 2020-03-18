

from glompo.generators.basegenerator import BaseGenerator
from typing import *
import numpy as np


__all__ = ("RandomGenerator",)


class RandomGenerator(BaseGenerator):
    """ Generates random starting points within given bounds. """

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        self.n_params = len(bounds)

        for bnd in bounds:
            if bnd[0] >= bnd[1]:
                raise ValueError("Invalid bounds encountered. Min and max bounds may not be equal nor may they be in"
                                 "the opposite order. ")
            if not np.all(np.isfinite(bnd)):
                raise ValueError("Non-finite bounds found.")
        self.bounds = np.array(bounds)

    def generate(self) -> np.ndarray:
        calc = (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]
        return calc
