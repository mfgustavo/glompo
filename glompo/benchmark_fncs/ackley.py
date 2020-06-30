

from time import sleep
import numpy as np
from typing import *
from ._base import BaseTestCase


class Ackley(BaseTestCase):
    """ When called returns evaluations of the Ackley function. """

    def __init__(self, dims: int = 2, delay: int = 0, a: float = 20, b: float = 0.2, c: float = 2*np.pi):
        """
        Implementation of the Ackley optimization test function.
        Recommended bounds: [-32.768, 32.768] * dims
        Global minimum: f(0, 0, ..., 0) = 0

        Multimodal flat surface with a single deep global minima. Multimodal version of the Easom function.

        Parameters
        ----------
        dims: int = 2
            Number of dimensions of the problem
        delay: int = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        a: float = 20
            Ackley function parameter
        b: float = 0.2
            Ackley function parameter
        c: float = 2*np.pi
            Ackley function parameter
        """
        self._dims = dims
        self._delay = delay
        self.a, self.b, self.c = a, b, c

    def __call__(self, x) -> float:
        x = np.array(x)
        term1 = -self.a
        sos = 1/self.dims * np.sum(x ** 2)
        term1 *= np.exp(-self.b * np.sqrt(sos))

        cos = 1/self.dims * np.sum(np.cos(self.c * x))
        term2 = -np.exp(cos)

        term34 = self.a + np.e

        sleep(self.delay)
        return term1+term2+term34

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-32.768, 32.768]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
