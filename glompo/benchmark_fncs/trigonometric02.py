from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Trigonometric(BaseTestCase):
    """ Implementation of the Trigonometric Type-II optimization test function [a]_.

        .. math::
           f(x) = 1 + \\sum_{i=1}^d 8 \\sin^2 \\left[7(x_i-0.9)^2\\right]
           + 6 \\sin^2 \\left[14(x_i-0.9)^2\\right]+(x_i-0.9)^2

        Recommended bounds: :math:`x_i \\in [-500, 500]`

        Global minimum: :math:`f(0.9, 0.9, ..., 0.9) = 1`

        .. image:: /_figs/trigonometric.png
           :align: center
           :alt: Parabolic but becomes a multimodal flat surface with many peaks and troughs near the minimum.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        core = (np.array(x) - 0.9) ** 2
        sin1 = 8 * np.sin(7 * core) ** 2
        sin2 = 6 * np.sin(14 * core) ** 2

        total = sin1 + sin2 + core
        return 1 + np.sum(total)

    @property
    def min_x(self) -> Sequence[float]:
        return [0.9] * self.dims

    @property
    def min_fx(self) -> float:
        return 1

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims
