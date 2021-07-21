from time import sleep
from typing import Sequence, Tuple

from ._base import BaseTestCase


class Rosenbrock(BaseTestCase):
    """ Implementation of the Rosenbrock optimization test function [b]_.

        .. math::
           f(x) = \\sum^{d-1}_{i=1}\\left[100(x_{i+1}-x_i^2)^2+(x_i-1)^2\\right]

        Recommended bounds: :math:`x_i \\in [-2.048, 2.048]`

        Global minimum: :math:`f(1, 1, ..., 1) = 0`

        .. image:: /_figs/rosenbrock.png
           :align: center
           :alt: Global minimum is located in a very easy to find valley but locating it within the valley is difficult.
    """

    def __call__(self, x):
        total = 0
        for i in range(self.dims - 1):
            total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        sleep(self.delay)
        return total

    @property
    def min_x(self) -> Sequence[float]:
        return [1] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-2.048, 2.048]] * self._dims
