

from time import sleep
import warnings
import numpy as np
from typing import *
from ._base import BaseTestCase


class Shekel(BaseTestCase):
    """ When called returns evaluations of the Shekel function. """

    def __init__(self, dims: int = 2, delay: int = 0, m: int = 10, shift_positive: bool = False):
        """
        Implementation of the Shekel optimization test function.
        Recommended bounds: [-32.768, 32.768] * dims
        Global minimum: f(4, 4, 4, 4) =~ -10

        Parameters
        ----------
        dims: int = 2
            Number of dimensions [1, 4] allowed.
        delay: int = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        m: int = 10
            Number of minima. Global minimum certified for m=5,7 and 10.
        shift_positive: bool
            Shifts the entire function such that the global minimum falls at 0.
            Since this is variable for this function the adjustment is +12 and thus the global minimum will not
            necessarily fall at zero.
        """
        assert 0 < dims < 5
        self._dims = dims
        self._delay = delay
        self.shift = shift_positive

        if any([m == i for i in (5, 7, 10)]):
            self.m = m

            self.a = np.array([[4] * 4,
                               [1] * 4,
                               [8] * 4,
                               [6] * 4,
                               [3, 7] * 2])
            self.c = 0.1 * np.array([1, 2, 2, 4])

            if m == 5:
                self.c = np.append(self.c, 0.6)
            else:
                self.a = np.append(self.a,
                                   np.array([[2, 9] * 2,
                                             [5, 5, 3, 3]]),
                                   axis=0)
                self.c = np.append(self.c, 0.1 * np.array([4, 6, 3]))
                if m == 10:
                    self.a = np.append(self.a,
                                       np.array([[8, 1] * 2,
                                                 [6, 2] * 2,
                                                 [7, 3.6] * 2]),
                                       axis=0)
                    self.c = np.append(self.c, 0.1 * np.array([7, 5, 5]))

        else:
            raise ValueError("m can only be 5, 7 or 10")

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)

        calc = (self.c + np.sum((x - self.a[:, :self.dims]) ** 2, axis=1)) ** -1
        calc = -np.sum(calc)

        if self.shift:
            return calc + 12

        return calc

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return [4] * self.dims

    @property
    def min_fx(self) -> float:
        warnings.warn("Global minimum is only known for some combinations of m and d. The provided value is "
                      "approximate.")
        return -10.6

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-32.768, 32.768]] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
