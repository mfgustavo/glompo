

from typing import *
import numpy as np
from .basegenerator import BaseGenerator
from ..common.helpers import is_bounds_valid
from ..common.namedtuples import Result


__all__ = ("IncumbentGenerator",)


class IncumbentGenerator(BaseGenerator):
    """ Starts a new optimizer at the current incumbent solution. A random vector is generated if this is
        indeterminate.
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        super().__init__()
        self.n_params = len(bounds)

        if is_bounds_valid(bounds):
            self.bounds = np.array(bounds)

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        best: Result = manager.result
        if best.x is not None:
            return (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]

        return best.x
