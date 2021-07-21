from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class StyblinskiTang(BaseTestCase):
    """ Implementation of the Styblinski-Tang optimization test function [b]_.

        .. math::
           f(x) = \\frac{1}{2}\\sum^d_{i=1}\\left(x_i^4-16x_i^2+5x_i\\right)

        Recommended bounds: :math:`x_i \\in [-500, 500]`

        Global minimum: :math:`f(-2.90, -2.90, ..., -2.90) = -39.16616570377 d`

        .. image:: /_figs/styblinskitang.png
           :align: center
           :alt: Similar to Qing function but minima are deceptively similar but not actually degenerate.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        calc = x ** 4 - 16 * x ** 2 + 5 * x

        return 0.5 * np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return [-2.903534018185960] * self.dims

    @property
    def min_fx(self) -> float:
        return -39.16616570377142 * self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims
