

from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Rastrigin(BaseTestCase):
    """ When called returns evaluations of the Rastrigin function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Rastrigin optimization test function.
        Recommended bounds: [-5.12, 5.12] * dims
        Global minimum: f(0, 0, ..., 0) = 0

        Modulation of a unimodal paraboloid with multiple regular local minima.

        Parameters
        ----------
        dims : int
            Number of dimensions of the function.
        delay : int
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        """
        self._dims = dims
        self._delay = delay

    def __call__(self, x):
        x = np.array(x)

        calc = 10 * self.dims
        calc += np.sum(x ** 2 - 10*np.cos(2*np.pi * x))

        sleep(self.delay)
        return calc

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
        return [[-5.12, 5.12]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
