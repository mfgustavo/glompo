""" Contains an abstract super class from which all other benchmark functions inherit. """

from abc import ABC, abstractmethod
from time import sleep
from typing import Sequence, Tuple


class BaseTestCase(ABC):
    """ Basic API for Optimization test cases. """

    def __init__(self, dims: int, *, delay: float = 0):
        """ Initialize function

            Parameters
            ----------
            dims : int
                Number of parameters in the input space.
            delay : float, default=0
                Pause (in seconds) between function evaluations to mimic slow functions.
        """
        self._dims = dims
        self._delay = delay

    @property
    def dims(self) -> int:
        """ Number of parameters in the input space. """
        return self._dims

    @property
    @abstractmethod
    def min_x(self) -> Sequence[float]:
        """ The location of the global minimum in parameter space. """

    @property
    @abstractmethod
    def min_fx(self) -> float:
        """ The function value of the global minimum. """

    @property
    @abstractmethod
    def bounds(self) -> Sequence[Tuple[float, float]]:
        """ Sequence of min/max pairs bounding the function in each dimension. """

    @property
    def delay(self) -> float:
        """ Delay (in seconds) between function evaluations to mimic slow functions. """
        return self._delay

    @abstractmethod
    def __call__(self, x: Sequence[float]) -> float:
        """ Evaluates the function.

            Parameters
            ----------
            x : list of float
                Vector in parameter space where the function will be evaluated.

            Returns
            -------
            float
                Function value at `x`.

        """
        sleep(self.delay)
