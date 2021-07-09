from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Alpine01(BaseTestCase):
    """ When called returns evaluations of the Alpine01 function. """

    def __init__(self, dims: int = 2, delay: float = 0):
        """
        Implementation of the Alpine01 optimization test function.
        Recommended bounds: [-10, 10] * dims
        Global minimum: f(0, 0, ..., 0) = 0
        Highly oscillatory non-periodic surface.

        Parameters
        ----------
        dims: int = 2
            Number of dimensions of the problem
        delay: float = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        """
        self._dims = dims
        self._delay = delay

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        calc = np.sin(x)
        calc *= x
        calc += 0.1 * x
        calc = np.abs(calc)

        return np.sum(calc)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-10, 10]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
