

""" Base generator from which all generators must inherit to be compatible with GloMPO. """

from typing import *
from abc import ABC, abstractmethod
import numpy as np


class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, *args, **kwargs) -> np.ndarray:
        """ Returns an array of parameters as a suggested starting point for an optimizer """
        pass

    def update(self, x: Sequence[float], fx: float):
        """ Optional method. If implemented then the class requires feedback from the optimizers in order to generate
            new suggestions.
        """
        pass
