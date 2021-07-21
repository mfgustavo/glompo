from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Levy(BaseTestCase):
    """ Implementation of the Levy optimization test function [b]_.

        .. math::
            f(x) & = & \\sin^2(\\pi w_1) + \\sum^{d-1}_{i=1}\\left(w_i-1\\right)^2\\left[1+10\\sin^2\\left(\\pi w_i +1
            \\right)\\right] + \\left(w_d-1\\right)^2\\left[1+\\sin^2\\left(2\\pi w_d\\right)\\right] \\\\
            w_i & = & 1 + \\frac{x_i - 1}{4}

        Recommended bounds: :math:`x_i \\in [-10, 10]`

        Global minimum: :math:`f(1, 1, ..., 1) = 0`

        .. image:: /_figs/levy.png
           :align: center
           :alt: Moderately oscillatory periodic surface.
    """

    def __call__(self, x: np.ndarray) -> float:
        x = np.array(x)
        w = self._w(x)

        term1 = np.sin(np.pi * w[0]) ** 2

        term2 = (w - 1) ** 2
        term2 *= 1 + 10 * np.sin(np.pi * w + 1) ** 2
        term2 = np.sum(term2)

        term3 = (w[-1] - 1) ** 2
        term3 *= 1 + np.sin(2 * np.pi * w[-1]) ** 2

        sleep(self.delay)
        return term1 + term2 + term3

    @staticmethod
    def _w(x: np.ndarray) -> np.ndarray:
        return 1 + (x - 1) / 4

    @property
    def min_x(self) -> Sequence[float]:
        return [1] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-10, 10]] * self.dims
