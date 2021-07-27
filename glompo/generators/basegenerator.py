import logging
from abc import ABC, abstractmethod

import numpy as np

__all__ = ("BaseGenerator",)


class BaseGenerator(ABC):
    """ Base generator from which all generators must inherit to be compatible with GloMPO. """

    def __init__(self):
        self.logger = logging.getLogger('glompo.generator')

    @abstractmethod
    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        """ Returns a vector representing a location in input space.
        The returned array serves as a starting point for an optimizer.

        Parameters
        ----------
        manager
            :class:`.GloMPOManager` instance which is managing the optimization. Its attributes can be accessed when
            determining the convergence criteria.
        """
