from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Alpine02(BaseTestCase):
    """ Implementation of the Alpine Type-II optimization test function [a]_.

        .. math::
           f(x) = - \\prod_{i=1}^n \\sqrt{x_i} \\sin{x_i}

        Recommended bounds: :math:`x_i \\in [0, 10]`

        Global minimum: :math:`f(7.917, 7.917, ..., 7.917) = -6.1295`

        .. image:: /_figs/alpine02.png
           :align: center
           :alt: Moderately oscillatory periodic surface.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        calc = np.sin(x)
        calc *= np.sqrt(x)

        return -np.prod(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return [7.917] * self.dims

    @property
    def min_fx(self) -> float:
        return -2.808 ** self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 10]] * self.dims
