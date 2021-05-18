from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Easom(BaseTestCase):
    """ When called returns evaluations of the Easom function. """

    def __init__(self, delay: float = 0, shift_positive: bool = False):
        """
        Implementation of the Easom optimization test function.
        Recommended bounds: [-100, 100] * 2
        Global minimum: f(pi, pi) = -1

        Totally flat surface with a single very small bullet hole type minimum.

        Parameters
        ----------
        delay : int
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        shift_positive: bool = False
            Shifts the entire function such that the global minimum falls at 0.
        """
        self._delay = delay
        self.shift = shift_positive

    def __call__(self, x):
        sleep(self.delay)

        calc = -np.cos(x[0])
        calc *= np.cos(x[1])
        calc *= np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)

        if self.shift:
            return calc + 1

        return calc

    @property
    def dims(self) -> int:
        return 2

    @property
    def min_x(self) -> Sequence[float]:
        return [np.pi, np.pi]

    @property
    def min_fx(self) -> float:
        return -1 if not self.shift else 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-100, 100]] * 2

    @property
    def delay(self) -> float:
        return self._delay
