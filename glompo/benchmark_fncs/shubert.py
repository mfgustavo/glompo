import warnings
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Shubert(BaseTestCase):
    """ Implementation of the Shubert Type-I, Type-III and Type-IV optimization test functions [a]_.

        .. math::
           f_I(x) & = & \\sum^2_{i=1}\\sum^5_{j=1} j \\cos\\left[(j+1)x_i+j\\right]\\\\
           f_{III}(x) & = & \\sum^5_{i=1}\\sum^5_{j=1} j \\sin\\left[(j+1)x_i+j\\right]\\\\
           f_{IV}(x) & = & \\sum^5_{i=1}\\sum^5_{j=1} j \\cos\\left[(j+1)x_i+j\\right]\\\\

        Recommended bounds: :math:`x_i \\in [-10, 10]`

        .. image:: /_figs/shubert.png
           :align: center
           :alt: Highly oscillatory, periodic surface. Many degenerate global minima regularly placed.
    """

    def __init__(self, dims: int = 2, style: int = 1, *, shift_positive: bool = False, delay: float = 0):
        """ Parameters
            ----------
            style: int = 1
                Selection between the Shubert01, Shubert03 & Shubert04 functions. Each more oscillatory than the previous.
            shift_positive: bool = False
                Shifts the entire function such that the global minimum falls at 0.
        """
        super().__init__(dims, delay=delay)
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
