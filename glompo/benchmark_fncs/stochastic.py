from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Stochastic(BaseTestCase):
    """ When called returns evaluations of the Stochastic function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Stochastic optimization test function.
        Recommended bounds: [-5, 5] * dims
        Global minimum: f(1/dims, 1/dims, ..., 1/dims) = 0
        Function with random evaluation noise making a substantial contribution to the function value.
        Generally parabolic in shape.

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
        x = np.array(x)
        e = np.random.rand(self.dims)
        i = np.arange(1, self.dims + 1)

        calc = e * np.abs(x - 1 / i)

        return np.sum(calc)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return 1 / np.arange(1, self.dims + 1)

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-5, 5]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
