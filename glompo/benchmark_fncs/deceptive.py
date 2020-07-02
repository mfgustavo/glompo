from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Deceptive(BaseTestCase):
    """ When called returns evaluations of the Deceptive function. """

    def __init__(self, dims: int = 2, delay: int = 0, b: float = 2, shift_positive: bool = False):
        """
        Implementation of the Deceptive optimization test function.
        Recommended bounds: [0, 1] * dims
        Global minimum: f(a) = -1

        Parameters
        ----------
        dims : int
            Number of dimensions of the function.
        delay : int
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        b: float = 2
            Non-linearity parameter.
        shift_positive: bool = False
            Shifts the entire function such that the global minimum falls at 0.
        """
        self._dims = dims
        self._delay = delay
        self.shift = shift_positive
        self.b = b
        self._min_x = np.random.uniform(0, 1, dims)

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)

        calc = - (1 / self.dims * np.sum(self.g(x))) ** self.b

        if self.shift:
            return calc + 1

        return calc

    def g(self, vec: np.ndarray):
        gx = np.zeros(len(vec))

        for i, x in enumerate(vec):
            ai = self._min_x[i]
            if 0 <= x <= 0.8 * ai:
                gx[i] = 0.8 - x / ai
            elif 0.8 * ai < x <= ai:
                gx[i] = 5 * x / ai - 4
            elif ai < x <= (1 + 4 * ai) / 5:
                gx[i] = (5 * (x - ai)) / (ai - 1) + 1
            elif (1 + 4 * ai) / 5 < x <= 1:
                gx[i] = (x - 1) / (1 - ai) + 0.8

        return gx

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return self._min_x

    @property
    def min_fx(self) -> float:
        return -1 if not self.shift else 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 1]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
