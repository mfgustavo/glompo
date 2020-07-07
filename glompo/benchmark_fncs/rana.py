from typing import Sequence, Tuple

import numpy as np

from ._base import BaseTestCase


class Rana(BaseTestCase):
    """ When called returns evaluations of the Rana function. """

    def __init__(self, dims: int = 2, delay: int = 0):
        """
        Implementation of the Rana optimization test function.
        Recommended bounds: [-500.000001, 500.000001] * dims
        Global minimum: f(-500, -500, ..., -500) = -928.5478
        Highly multimodal and chaotic, optimum is on the lower bound

        Parameters
        ----------
        dims: int = 2
            Number of dimensions of the problem
        delay: int = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        """
        self._dims = dims
        self._delay = delay

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
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [-500] * self.dims

    @property
    def min_fx(self) -> float:
        return self.__call__(self.min_x)

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500.000001, 500.000001]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
