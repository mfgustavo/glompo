from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Ackley(BaseTestCase):
    """ Implementation of the Ackley optimization test function [b]_.

        .. math::
           f(x) = - a \\exp\\left(-b \\sqrt{\\frac{1}{d}\\sum^d_{i=1}x_i^2}\\right)
                  - \\exp\\left(\\frac{1}{d}\\sum^d_{i=1}\\cos\\left(cx_i\\right)\\right)
                  + a
                  + \\exp(1)

        Recommended bounds: :math:`x_i \\in [-32.768, 32.768]`

        Global minimum: :math:`f(0, 0, ..., 0) = 0`

        .. image:: /_figs/ackley.png
           :align: center
           :alt: Multimodal flat surface with a single deep global minima. Multimodal version of the Easom function.
    """

    def __init__(self, dims: int = 2, a: float = 20, b: float = 0.2, c: float = 2 * np.pi, *, delay: float = 0):
        """ Parameters
            ----------
            a : float=20
                Ackley function parameter
            b : float=0.2
                Ackley function parameter
            c : float=2*np.pi
                Ackley function parameter
        """
        super().__init__(dims, delay=delay)
        self.a, self.b, self.c = a, b, c

    def __call__(self, x) -> float:
        x = np.array(x)
        term1 = -self.a
        sos = 1 / self.dims * np.sum(x ** 2)
        term1 *= np.exp(-self.b * np.sqrt(sos))

        cos = 1 / self.dims * np.sum(np.cos(self.c * x))
        term2 = -np.exp(cos)

        term34 = self.a + np.e

        sleep(self.delay)
        return term1 + term2 + term34

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [(-32.768, 32.768)] * self.dims
