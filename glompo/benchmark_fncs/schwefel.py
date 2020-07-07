from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Schwefel(BaseTestCase):
    """ When called returns evaluations of the Schwefel function. """

    def __init__(self, dims: int = 2, delay: int = 0, shift_positive: bool = False):
        """
        Implementation of the Schwefel optimization test function.
        Recommended bounds: [-500, 500] * dims
        Global minimum: f(420.9687, 420.9687, ..., 420.9687) = -418.9829d

        Multimodal and deceptive in that the global minimum is very far from the next best local
        minimum

        Parameters
        ----------
        dims : int
            Number of dimensions of the function.
        delay : int
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        shift_positive: bool = False
            Shifts the entire function such that the global minimum falls at ~0.
        """
        self._dims = dims
        self._delay = delay
        self.shift = shift_positive

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)
        calc = np.sum(-x * np.sin(np.sqrt(np.abs(x))))

        if self.shift:
            return calc + 418.9830 * self.dims

        return calc

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [420.9687] * self.dims

    @property
    def min_fx(self) -> float:
        return -418.9829 * self.dims if not self.shift else 0.0001 * self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
