

from typing import *
import numpy as np
from .basegenerator import BaseGenerator
from ..common.helpers import is_bounds_valid


__all__ = ("IncumbentGenerator",)


class IncumbentGenerator(BaseGenerator):
    """ Starts a new optimizer at the current incumbent solution. A random vector is generated if this is
        indeterminate.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        self.best_fx = np.inf
        self.best_x = None
        self.n_params = len(bounds)

        if is_bounds_valid(bounds):
            self.bounds = np.array(bounds)

    def generate(self) -> np.ndarray:
        if self.best_x is None:
            return (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]
        return self.best_x

    def update(self, x: Sequence[float], fx: float):
        if fx < self.best_fx:
            self.best_fx = fx
            self.best_x = x
