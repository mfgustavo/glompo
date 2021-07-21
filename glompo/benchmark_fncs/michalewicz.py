import warnings
from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Michalewicz(BaseTestCase):
    """ Implementation of the Michalewicz optimization test function [b]_.

        .. math::
            f(x) = - \\sum^d_{i=1}\\sin(x_i)\\sin^{2m}\\left(\\frac{ix_i^2}{\\pi}\\right)

        Recommended bounds: :math:`x_i \\in [0, \\pi]`

        Global minimum:

        .. math::

            f(x) = \\begin{cases}
                        -1.8013 & \\text{if} & d=2 \\\\
                        -4.687 & \\text{if} & d=5 \\\\
                        -9.66 & \\text{if} & d=10 \\\\
                   \\end{cases}

        .. image:: /_figs/michalewicz.png
           :align: center
           :alt: Flat surface with many valleys and a single global minimum.

    """

    def __init__(self, dims: int = 2, m: float = 10, *, delay: float = 0):
        """ Parameters
            ----------
            m : float
                Parametrization of the function. Lower values make the valleys more informative at pointing to the
                minimum. High values (:math:`\\pm10`) create a needle-in-a-haystack function where there is no
                information pointing to the minimum.
        """
        super().__init__(dims, delay=delay)
        self.m = m

    def __call__(self, x):
        sleep(self.delay)

        i = np.arange(1, len(x) + 1)
        x = np.array(x)

        calc = np.sin(x)
        calc *= np.sin(i * x ** 2 / np.pi) ** (2 * self.m)

        return -np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Global minimum only known for d=2, 5 and 10 but locations are unknown.")
        return [0] * self._dims

    @property
    def min_fx(self) -> float:
        if self._dims == 2:
            return -1.8013
        if self._dims == 5:
            return -4.687658
        if self._dims == 10:
            return -9.66015
        warnings.warn("Global minimum only known for d=2, 5 and 10")
        return np.inf

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [(0, np.pi)] * self.dims
