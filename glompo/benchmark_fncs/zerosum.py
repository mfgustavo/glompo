from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class ZeroSum(BaseTestCase):
    """ When called returns evaluations of the Zero Sum function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Zero Sum optimization test function.
        Recommended bounds: [-10, 10] * dims
        Global minimum: f(x1 + x2 + x3 ... + xn = 0 ) = 0

        Single valley of degenerate global minimum results that is not axi-parallel

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

        tot = np.sum(x)

        if tot == 0:
            return 0

        calc = np.abs(tot)
        calc *= 10000
        calc **= 0.5
        calc += 1

        return calc

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [7.70628098] * self.dims

    @property
    def min_fx(self) -> float:
        return -self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0.25, 10]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
