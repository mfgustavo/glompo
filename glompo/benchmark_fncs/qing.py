import warnings
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Qing(BaseTestCase):
    """ Implementation of the Qing optimization test function [a]_.

        .. math::
           f(x) = \\sum^d_{i=1} (x_i^2-i)^2

        Recommended bounds: :math:`x_i \\in [-500, 500]`

        Global minimum: :math:`f(\\sqrt{1}, \\sqrt{2}, ..., \\sqrt{n}) = 0`

        .. image:: /_figs/qing.png
           :align: center
           :alt: Globally flat with parabolic walls but has :math:`2^d` degenerate global minima.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)
        i = np.arange(1, self.dims + 1)

        calc = (x ** 2 - i) ** 2

        return np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Global minimum is degenerate at every positive and negative combination of the returned "
                      "parameter vector.", UserWarning)
        return np.sqrt(np.arange(1, self.dims + 1))

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims
