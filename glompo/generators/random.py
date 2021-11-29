import numpy as np

from .basegenerator import BaseGenerator

__all__ = ("RandomGenerator",)


class RandomGenerator(BaseGenerator):
    """ Generates random points.
    Points are drawn from a uniform distribution within given `bounds`.
    """
    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        bounds = np.array(manager.bounds)
        n_parms = manager.n_parms

        calc = (bounds[:, 1] - bounds[:, 0]) * np.random.random(n_parms) + bounds[:, 0]
        return calc
