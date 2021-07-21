from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Easom(BaseTestCase):
    """ Implementation of the Easom optimization test function [a]_.

        .. math::
           f(x) = - \\cos\\left(x_1\\right)\\cos\\left(x_2\\right)\\exp\\left(-(x_1-\\pi)^2-(x_2-\\pi)^2\\right)

        Recommended bounds: :math:`x_1,x _2 \\in [-100, 100]`

        Global minimum: :math:`f(\\pi, \\pi) = -1`

        .. image:: /_figs/easom.png
           :align: center
           :alt: Totally flat surface with a single very small bullet hole type minimum.
    """

    def __init__(self, *args, shift_positive: bool = False, delay: float = 0):
        """ Parameters
            ----------
            shift_positive : bool, default=False
                Shifts the entire function such that the global minimum falls at 0.
        """
        super().__init__(2, delay=delay)
        self.shift = shift_positive

    def __call__(self, x):
        sleep(self.delay)

        calc = -np.cos(x[0])
        calc *= np.cos(x[1])
        calc *= np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)

        if self.shift:
            return calc + 1

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [np.pi, np.pi]

    @property
    def min_fx(self) -> float:
        return -1 if not self.shift else 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-100, 100]] * 2
