from typing import Sequence, Tuple

import numpy as np

from .basegenerator import BaseGenerator
from ..common.helpers import is_bounds_valid

__all__ = ("RandomGenerator",)


class RandomGenerator(BaseGenerator):
    """ Generates random points
    Points are drawn from a uniform distribution within given `bounds`.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        super().__init__()
        self.n_params = len(bounds)
        if is_bounds_valid(bounds):
            self.bounds = np.array(bounds)

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        calc = (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]
        return calc
