

from time import sleep
import numpy as np
from typing import *
from ._base import BaseTestCase


class Levy(BaseTestCase):
    """ When called returns evaluations of the Levy function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Levy optimization test function.
        Recommended bounds: [-10, 10] * dims
        Global minimum: f(1, 1, ..., 1) = 0

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

    def __call__(self, x: np.ndarray) -> float:
        x = np.array(x)
        w = self._w(x)

        term1 = np.sin(np.pi * w[0]) ** 2

        term2 = (w - 1) ** 2
        term2 *= 1 + 10*np.sin(np.pi * w + 1) ** 2
        term2 = np.sum(term2)

        term3 = (w[-1] - 1) ** 2
        term3 *= 1 + np.sin(2 * np.pi * w[-1]) ** 2

        sleep(self.delay)
        return term1 + term2 + term3

    @staticmethod
    def _w(x: np.ndarray) -> np.ndarray:
        return 1 + (x - 1) / 4

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
        return [[-10, 10]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
