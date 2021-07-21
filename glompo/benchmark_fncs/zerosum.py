from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class ZeroSum(BaseTestCase):
    """ Implementation of the ZeroSum optimization test function [a]_.

        .. math::
           f(x) = \\begin{cases}
                        0 & \text{if} \\sum^n_{i=1} x_i = 0 \\\\
                        1 + (10000|\\sum^n_{i=1} x_i = 0|)^{0.5} & \text{otherwise}
                  \\end{cases}

        Recommended bounds: :math:`x_i \\in [-10, 10]`

        Global minimum: :math:`f(x) = 0 \text{where} \\sum^n_{i=1} x_i = 0`

        .. image:: /_figs/zerosum.png
           :align: center
           :alt: Single valley of degenerate global minimum results that is not axi-parallel.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        tot = np.sum(x)

        if tot == 0:
            return 0

        calc = np.abs(tot)
        calc *= 10000
        calc **= 0.5
        calc += 1

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [7.70628098] * self.dims

    @property
    def min_fx(self) -> float:
        return -self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-10, 10]] * self.dims
