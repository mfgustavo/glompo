from time import sleep
from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Schwefel(BaseTestCase):
    """ Implementation of the Schwefel optimization test function [b]_.

        .. math::
           f(x) = 418.9829d - \\sum^d_{i=1} x_i\\sin\\left(\\sqrt{|x_i|}\\right)

        Recommended bounds: :math:`x_i \\in [-500, 500]`

        Global minimum: :math:`f(420.9687, 420.9687, ..., 420.9687) = -418.9829d`

        .. image:: /_figs/schwefel.png
           :align: center
           :alt: Multimodal and deceptive in that the global minimum is very far from the next best local minimum.
    """

    def __init__(self, dims: int = 2, *, shift_positive: bool = False, delay: float = 0):
        """ Parameters
            ----------
            shift_positive : bool, default=False
                Shifts the entire function such that the global minimum falls at ~0.
        """
        super().__init__(dims, delay=delay)
        self.shift = shift_positive

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)
        calc = np.sum(-x * np.sin(np.sqrt(np.abs(x))))

        if self.shift:
            return calc + 418.9830 * self.dims

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [420.9687] * self.dims

    @property
    def min_fx(self) -> float:
        return -418.9829 * self.dims if not self.shift else 0.0001 * self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims
