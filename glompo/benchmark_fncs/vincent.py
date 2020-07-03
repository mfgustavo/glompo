from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Vincent(BaseTestCase):
    """ When called returns evaluations of the Vincent function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Vincent optimization test function.
        Recommended bounds: [0.25, 10] * dims
        Global minimum: f(7.706, 7.706, ..., 7.706) = -dims

        'Flat' surface made of period peaks and trough of various sizes.

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

    def __call__(self, x) -> float:
        super().__call__(x)

        calc = 10 * np.log(x)
        calc = np.sin(calc)
        calc = -np.sum(calc)

        return calc

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [7.70628098] * self.dims

    @property
    def min_fx(self) -> float:
        return -self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0.25, 10]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
