import warnings
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Qing(BaseTestCase):
    """ When called returns evaluations of the Qing function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Qing optimization test function.
        Recommended bounds: [-500, 500] * dims
        Global minimum: f(sqrt(1), sqrt(2), ..., sqrt(n)) = 0
        Globally flat with parabolic walls but has 2^dims degenerate global minima.

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
        x = np.array(x)
        i = np.arange(1, self.dims + 1)

        calc = (x ** 2 - i) ** 2

        return np.sum(calc)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Global minimum is degenerate at every positive and negative combination of the returned "
                      "parameter vector.", UserWarning)
        return np.sqrt(np.arange(1, self.dims + 1))

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
