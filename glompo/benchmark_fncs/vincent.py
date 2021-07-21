from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Vincent(BaseTestCase):
    """ Implementation of the Vincent optimization test function [a]_.

        .. math::
           f(x) = - \\sum^d_{i=1} \\sin\\left(10\\log(x)\\right)

        Recommended bounds: :math:`x_i \\in [0.25, 10]`

        Global minimum: :math:`f(7.706, 7.706, ..., 7.706) = -d`

        .. image:: /_figs/vincent.png
           :align: center
           :alt: 'Flat' surface made of period peaks and trough of various sizes.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        calc = 10 * np.log(x)
        calc = np.sin(calc)
        calc = -np.sum(calc)

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [7.70628098] * self.dims

    @property
    def min_fx(self) -> float:
        return -self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0.25, 10]] * self.dims
