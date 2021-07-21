from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Griewank(BaseTestCase):
    """ Implementation of the Griewank optimization test function [b]_.

        .. math::
           f(x) = \\sum_{i=1}^d \\frac{x_i^2}{4000} - \\prod_{i=1}^d \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1

        Recommended bounds: :math:`x_i \\in [-600, 600]`

        Global minimum: :math:`f(0, 0, ..., 0) = 0`

        .. image:: /_figs/griewank.png
           :align: center
           :alt: Highly oscillatory totally-periodic surface on a general parabolic surface. Similar to Rastrigin.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        term1 = 1 / 4000 * np.sum(x ** 2)

        term2 = x / np.sqrt(np.arange(1, len(x) + 1))
        term2 = np.prod(np.cos(term2))

        return 1 + term1 - term2

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-600, 600]] * self.dims
