from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class StyblinskiTang(BaseTestCase):
    """ When called returns evaluations of the Styblinski-Tang function. """

    def __init__(self, dims: int = 2, delay: float = 0):
        """
        Implementation of the Qing optimization test function.
        Recommended bounds: [-500, 500] * dims
        Global minimum: f(-2.90, -2.90, ..., -2.90) = -39.16616570377 * dims
        Similar to Qing function but minima are deceptively similar but not actually degenerate

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

        calc = x ** 4 - 16 * x ** 2 + 5 * x

        return 0.5 * np.sum(calc)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [-2.903534018185960] * self.dims

    @property
    def min_fx(self) -> float:
        return -39.16616570377142 * self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
