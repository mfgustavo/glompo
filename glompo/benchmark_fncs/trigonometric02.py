from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Trigonometric(BaseTestCase):
    """ When called returns evaluations of the Trigonometric02 function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Trigonometric02 optimization test function.
        Recommended bounds: [-500, 500] * dims
        Global minimum: f(0.9, 0.9, ..., 0.9) = 1

        Looks like a paraboloid but becomes a multimodal flat surface with many peaks and troughs near the minimum.

        Parameters
        ----------
        dims: int = 2
            Number of dimensions of the problem
        delay: int = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        """
        self._dims = dims
        self._delay = delay

    def __call__(self, x) -> float:
        super().__call__(x)

        core = (np.array(x) - 0.9) ** 2
        sin1 = 8 * np.sin(7 * core) ** 2
        sin2 = 6 * np.sin(14 * core) ** 2

        total = sin1 + sin2 + core
        return 1 + np.sum(total)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [0.9] * self.dims

    @property
    def min_fx(self) -> float:
        return 1

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
