from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Alpine01(BaseTestCase):
    """ Implementation of the Alpine Type-I optimization test function [a]_.

        .. math::
           f(x) = \\sum^n_{i=1}\\left|x_i\\sin\\left(x_i\\right)+0.1x_i\\right|

        Recommended bounds: :math:`x_i \\in [-10, 10]`

        Global minimum: :math:`f(0, 0, ..., 0) = 0`

        .. image:: /_figs/alpine01.png
           :align: center
           :alt: Highly oscillatory non-periodic surface.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        calc = np.sin(x)
        calc *= x
        calc += 0.1 * x
        calc = np.abs(calc)

        return np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [(-10, 10)] * self.dims
