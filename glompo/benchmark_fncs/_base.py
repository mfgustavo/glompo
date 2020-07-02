from abc import ABC, abstractmethod
from time import sleep
from typing import Sequence, Tuple


class BaseTestCase(ABC):
    """ Basic API for Optimization test cases. """

    @property
    @abstractmethod
    def dims(self) -> int:
        pass

    @property
    @abstractmethod
    def min_x(self) -> Sequence[float]:
        pass

    @property
    @abstractmethod
    def min_fx(self) -> float:
        pass

    @property
    @abstractmethod
    def bounds(self) -> Sequence[Tuple[float, float]]:
        pass

    @property
    @abstractmethod
    def delay(self) -> float:
        pass

    @abstractmethod
    def __call__(self, x) -> float:
        sleep(self.delay)
