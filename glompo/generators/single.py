

from typing import *
import numpy as np
from .basegenerator import BaseGenerator
from ..common.helpers import is_bounds_valid


__all__ = ("SinglePointGenerator",)


class SinglePointGenerator(BaseGenerator):
    """ Always returns the same point. Either provided during initialisation or otherwise randomly generated. """

    def __init__(self, bounds: Sequence[Tuple[float, float]], x: Optional[Sequence[float]] = None):
        self.n_params = len(bounds)
        if is_bounds_valid(bounds):
            self.bounds = np.array(bounds)

        if x is not None:
            self.vector = x
        else:
            self.vector = (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]

    def generate(self) -> np.ndarray:
        return self.vector
