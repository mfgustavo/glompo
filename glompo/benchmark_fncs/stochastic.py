from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Stochastic(BaseTestCase):
    """ Implementation of the Stochastic optimization test function [a]_.

        .. math::
           f(x) & = & \\sum^d_{i=1} \\epsilon_i\\left|x_i-\\frac{1}{i}\\right| \\\\
           \\epsilon_i & = & \\mathcal{U}_{[0, 1]}

        Recommended bounds: :math:`x_i \\in [-5, 5]`

        Global minimum: :math:`f(1/d, 1/d, ..., 1/d) = 0`

        .. image:: /_figs/stochastic.png
           :align: center
           :alt: Parabolic function with random evaluation noise making a substantial contribution.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)
        e = np.random.rand(self.dims)
        i = np.arange(1, self.dims + 1)

        calc = e * np.abs(x - 1 / i)

        return np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return 1 / np.arange(1, self.dims + 1)

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-5, 5]] * self.dims
