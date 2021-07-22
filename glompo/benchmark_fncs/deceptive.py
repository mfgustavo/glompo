from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Deceptive(BaseTestCase):
    """ Implementation of the Deceptive optimization test function [a]_.

        .. math:
           f(x) = - \\left[\\frac{1}{n}\\sum^n_{i=1}g_i\\left(x_i\\right)\\right]

        Recommended bounds: :math:`x_i \\in [0, 1]`

        Global minimum: :math:`f(a) = -1`

        .. image:: /_figs/deceptive.png
           :align: center
           :alt: Small global minimum surrounded by areas which slope away from it.
    """

    def __init__(self, dims: int = 2, b: float = 2, *, shift_positive: bool = False, delay: float = 0):
        """ Parameters
            ----------
            b : float, default=2
                Non-linearity parameter.
            shift_positive : bool, default=False
                Shifts the entire function such that the global minimum falls at 0.
        """
        super().__init__(dims, delay=delay)
        self.shift = shift_positive
        self.b = b
        self._min_x = np.random.uniform(0, 1, dims)

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)

        calc = - (1 / self.dims * np.sum(self._g(x))) ** self.b

        if self.shift:
            return calc + 1

        return calc

    def _g(self, vec: np.ndarray):
        """ Sub-calculation of the :meth:`__call__` method which returns :math:`g(x)`. """
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
    def min_x(self) -> Sequence[float]:
        return self._min_x

    @property
    def min_fx(self) -> float:
        return -1 if not self.shift else 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 1]] * self.dims
