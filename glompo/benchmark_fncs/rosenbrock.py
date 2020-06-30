

from time import sleep
from ._base import BaseTestCase
from typing import *


class Rosenbrock(BaseTestCase):
    """ When called returns evaluations of the Rosenbrock function. """

    def __init__(self, dims: int, delay: int = 0):
        """
        Implementation of the Rosenbrock optimization test function.
        Recommended bounds: [-2.048, 2.048] * dims
        Global minimum: f(1, 1, ..., 1) = 0

        Global minimum is located in a very easy to find valley but locating it within the valley is difficult.

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
        total = 0
        for i in range(self.dims-1):
            total += 100 * (x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        sleep(self.delay)
        return total

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [1] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-2.048, 2.048]] * self._dims

    @property
    def delay(self) -> float:
        return self._delay
