

""" Base generator from which all generators must inherit to be compatible with GloMPO. """

from typing import *
from abc import ABC, abstractmethod
import logging

import numpy as np


__all__ = ("BaseGenerator",)


class BaseGenerator(ABC):

    def __init__(self):
        self.logger = logging.getLogger('glompo.generator')

    @abstractmethod
    def generate(self) -> np.ndarray:
        """ Returns an array of parameters as a suggested starting point for an optimizer """

    def update(self, x: Sequence[float], fx: float):
        """ Optional method. If implemented then the class requires feedback from the optimizers in order to generate
            new suggestions.
        """
