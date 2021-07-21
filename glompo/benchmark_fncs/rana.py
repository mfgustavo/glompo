from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Rana(BaseTestCase):
    """ Implementation of the Rana optimization test function [a]_.

        .. math::
           f(x) = \\sum^d_{i=1}\\left[x_i\\sin\\left(\\sqrt{\\left|x_1-x_i+1\\right|}\\right)
                                      \\cos\\left(\\sqrt{\\left|x_1+x_i+1\\right|}\\right)\\\\
                                 + (x_1+1)\\sin\\left(\\sqrt{\\left|x_1+x_i+1\\right|}\\right)
                                      \\cos\\left(\\sqrt{\\left|x_1-x_i+1\\right|}\\right)
                              \\right]

        Recommended bounds: :math:`x_i \\in [-500.000001, 500.000001]`

        Global minimum: :math:`f(-500, -500, ..., -500) = -928.5478`

        .. image:: /_figs/rana.png
           :align: center
           :alt: Highly multimodal and chaotic, optimum is on the lower bound
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        term1 = x
        term1 = term1 * np.sin(np.sqrt(np.abs(x[0] - x + 1)))
        term1 = term1 * np.cos(np.sqrt(np.abs(x[0] + x + 1)))

        term2 = x[0] + 1
        term2 = term2 * np.sin(np.sqrt(np.abs(x[0] + x + 1)))
        term2 = term2 * np.cos(np.sqrt(np.abs(x[0] - x + 1)))

        return np.sum(term1 + term2)

    @property
    def min_x(self) -> Sequence[float]:
        return [-500] * self.dims

    @property
    def min_fx(self) -> float:
        return self.__call__(self.min_x)

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500.000001, 500.000001]] * self.dims
