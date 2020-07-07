""" Base generator from which all generators must inherit to be compatible with GloMPO. """

import logging
from abc import ABC, abstractmethod

import numpy as np

__all__ = ("BaseGenerator",)


class BaseGenerator(ABC):

    def __init__(self):
        self.logger = logging.getLogger('glompo.generator')

    @abstractmethod
    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        """ Returns an array of parameters as a suggested starting point for an optimizer.
            The manager itself is provided for access to iteration histories etc.
        """
