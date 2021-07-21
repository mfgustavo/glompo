from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Langermann(BaseTestCase):
    """ When called returns evaluations of the Langermann function [a]_ [b]_.

        .. math::
          f(x) & = & - \\sum_{i=1}^5 \\frac{c_i\\cos\\left(\\pi\\left[(x_1-a_i)^2 + (x_2-b_i)^2\\right]\\right)}
                                           {\\exp\\left(\\frac{(x_1-a_i)^2 + (x_2-b_i)^2}{\\pi}\\right)}\\\\
          \\mathbf{a} & = & \\{3, 5, 2, 1, 7\\}\\\\
          \\mathbf{b} & = & \\{5, 2, 1, 4, 9\\}\\\\
          \\mathbf{c} & = & \\{1, 2, 5, 2, 3\\}\\\\

        Recommended bounds: :math:`x_1, x_2 \\in [0, 10]`

        Global minimum: :math:`f(2.00299219, 1.006096) = -5.1621259`

        .. image:: /_figs/langermann.png
           :align: center
           :alt: Analogous to ripples on a water surface after three drops have hit it.
    """

    def __init__(self, *args, shift_positive: bool = False, delay: float = 0):
        """ Parameters
            ----------
            shift_positive : bool, default=False
                Shifts the entire function such that the global minimum falls at ~0.
        """
        super().__init__(2, delay=delay)
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
    def min_x(self) -> Sequence[float]:
        return [2.00299219, 1.006096]

    @property
    def min_fx(self) -> float:
        return -5.1621259 if not self.shift else 0.0378741

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 10]] * 2
