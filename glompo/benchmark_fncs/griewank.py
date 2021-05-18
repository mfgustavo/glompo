from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Griewank(BaseTestCase):
    """ When called returns evaluations of the Griewank function. """

    def __init__(self, dims: int = 2, delay: float = 0):
        """
        Implementation of the Griewank optimization test function.
        Recommended bounds: [-600, 600] * dims
        Global minimum: f(0, 0, ..., 0) = 0
        Highly oscillatory totally-periodic surface on a general parabolic surface. Similar to Rastrigin.

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

        term1 = 1 / 4000 * np.sum(x ** 2)

        term2 = x / np.sqrt(np.arange(1, len(x) + 1))
        term2 = np.prod(np.cos(term2))

        return 1 + term1 - term2

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
        return [[-600, 600]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
