from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Alpine02(BaseTestCase):
    """ When called returns evaluations of the Alpine02 function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Alpine02 optimization test function.
        Recommended bounds: [0, 10] * dims
        Global minimum: f(7.917, 7.917, ..., 7.917) = -6.1295
        Moderately oscillatory periodic surface.

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

        calc = np.sin(x)
        calc *= np.sqrt(x)

        return -np.prod(calc)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [7.917] * self.dims

    @property
    def min_fx(self) -> float:
        return -2.808 ** self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 10]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
