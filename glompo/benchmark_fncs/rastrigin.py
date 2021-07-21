from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Rastrigin(BaseTestCase):
    """ Implementation of the Rastrigin optimization test function [b]_.

        .. math::
           f(x) = 10d + \\sum^d_{i=1} \\left[x_i^2-10\\cos(2\\pi x_i)\\right]

        Recommended bounds: :math:`x_i \\in [-5.12, 5.12]`

        Global minimum: :math:`f(0, 0, ..., 0) = 0`

        .. image:: /_figs/rastrigin.png
           :align: center
           :alt: Modulation of a unimodal paraboloid with multiple regular local minima.
    """

    def __call__(self, x):
        x = np.array(x)

        calc = 10 * self.dims
        calc += np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

        sleep(self.delay)
        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-5.12, 5.12]] * self.dims
