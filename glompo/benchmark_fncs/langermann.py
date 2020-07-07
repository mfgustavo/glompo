from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Langermann(BaseTestCase):
    """ When called returns evaluations of the Langermann function. """

    def __init__(self, delay: int = 0, shift_positive: bool = False):
        """
        Implementation of the Levy optimization test function.
        Recommended bounds: [0, 10] * 2
        Global minimum: f(2.00299219, 1.006096) = -5.1621259

        Analogous to ripples on a water surface after three drops have hit it.

        Parameters
        ----------
        delay: int = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        shift_positive: bool = False
            Shifts the entire function such that the global minimum falls at ~0.
        """
        self._delay = delay
        self.shift = shift_positive

    def __call__(self, x: np.ndarray) -> float:
        a = np.array([3, 5, 2, 1, 7])
        b = np.array([5, 2, 1, 4, 9])
        c = np.array([1, 2, 5, 2, 3])
        x1, x2 = x[0], x[1]

        cos = c * np.cos(np.pi * ((x1 - a) ** 2 + (x2 - b) ** 2))
        exp = np.exp(((x1 - a) ** 2 + (x2 - b) ** 2) / np.pi)

        sleep(self.delay)
        calc = - np.sum(cos / exp)

        if self.shift:
            return calc + 5.2

        return calc

    @property
    def dims(self) -> int:
        return 2

    @property
    def min_x(self) -> Sequence[float]:
        return [2.00299219, 1.006096]

    @property
    def min_fx(self) -> float:
        return -5.1621259 if not self.shift else 0.0378741

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 10]] * 2

    @property
    def delay(self) -> float:
        return self._delay
