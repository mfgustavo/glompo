

import warnings
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Shubert(BaseTestCase):
    """ When called returns evaluations of the Shubert01, Shubert03 or Shubert04 functions. """

    def __init__(self, dims: int = 2, delay: float = 0, style: int = 1, shift_positive: bool = False):
        """
        Implementation of the Shubert optimization test functions.
        Recommended bounds: [-10, 10] * dims
        Global minimum is degenerate in all styles:
         - Shubert01 f_min = -186.7309
         - Shubert03 f_min = -24.062499
         - Shubert04 f_min = -29.016015

        Highly oscillatory and periodic surface with high peaks and deep narrow troughs. The global minima is
        degenerate at many locations but each is tightly surrounded by similar local minima.

        Parameters
        ----------
        dims: int = 2
            Number of dimensions of the problem
        delay: int = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        style: int = 1
            Selection between the Shubert01, Shubert03 & Shubert04 functions. Each more oscillatory than the previous.
        shift_positive: bool = False
            Shifts the entire function such that the global minimum falls at 0.
        """
        self._dims = dims
        self._delay = delay
        self.style = style
        self.shift_positive = shift_positive

    def __call__(self, x) -> float:
        super().__call__(x)

        if self.style == 1:
            x = np.array(x).reshape((-1, 1))
            j = np.arange(1, 6)

            calc = (j + 1) * x + j
            calc = np.cos(calc)
            calc = j * calc
            calc = np.sum(calc, axis=1)
            calc = np.prod(calc)

            if self.shift_positive:
                calc += 186.731

        elif self.style == 3:
            x = np.reshape(x, (-1, 1))
            j = np.arange(1, 6)

            calc = (j + 1) * x + j
            calc = np.sin(calc)
            calc = j * calc
            calc = np.sum(calc)

            if self.shift_positive:
                calc += 24.062499

        else:
            x = np.reshape(x, (-1, 1))
            j = np.arange(1, 6)

            calc = (j + 1) * x + j
            calc = np.cos(calc)
            calc = j * calc
            calc = np.sum(calc)

            if self.shift_positive:
                calc += 29.016015

        return calc

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Degenerate global minima")
        return

    @property
    def min_fx(self) -> float:
        if self._dims > 2:
            warnings.warn("Minimum unknown for d>2")
            return None

        if self.shift_positive:
            return 0

        if self.style == 1:
            return -186.7309
        if self.style == 3:
            return -24.062499
        return -29.016015

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-10, 10]] * self._dims

    @property
    def delay(self) -> float:
        return self._delay
