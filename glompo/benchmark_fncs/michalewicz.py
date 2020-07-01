

import warnings
from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Michalewicz(BaseTestCase):
    """ When called returns evaluations of the Michalewicz function. """

    def __init__(self, dims: int = 2, delay: int = 0, m: float = 10):
        """
        Implementation of the Michalewicz optimization test function.
        Recommended bounds: [0, pi] * dims
        Global minimum:
            f(x) = -1.8013 for d=2
            f(x) = -4.687 for d=5
            f(x) = -9.66 for d=10

        Flat surface with many valleys and a single global minimum.

        Parameters
        ----------
        dims : int
            Number of dimensions of the function.
        delay : int
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        m: float
            Parameterization of the function. Lower values make the valleys more informative at pointing to the minimum.
            High values (+-10) create a needle-in-a-haystack function where there is no information pointing to the
            minimum.
        """
        self._dims = dims
        self._delay = delay
        self.m = m

    def __call__(self, x):
        sleep(self.delay)

        i = np.arange(1, len(x)+1)
        x = np.array(x)

        calc = np.sin(x)
        calc *= np.sin(i*x**2 / np.pi) ** (2*self.m)

        return -np.sum(calc)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Global minimum only known for d=2, 5 and 10 but locations are unknown.")
        return [0] * self._dims

    @property
    def min_fx(self) -> float:
        if self._dims == 2:
            return -1.8013
        if self._dims == 5:
            return -4.687658
        if self._dims == 10:
            return -9.66015
        warnings.warn("Global minimum only known for d=2, 5 and 10")
        return np.inf

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, np.pi]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
