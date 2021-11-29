import numpy as np

from .basegenerator import BaseGenerator
from ..common.namedtuples import Result

__all__ = ("IncumbentGenerator",)


class IncumbentGenerator(BaseGenerator):
    """ Starts a new optimizer at :attr:`GloMPOManager.result['x'] <.GloMPOManager.result>`.
    A random vector is generated if this is indeterminate.
    """
    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        best: Result = manager.result

        bounds = np.array(manager.bounds)
        n_parms = manager.n_parms
        if best.x is None:
            return (bounds[:, 1] - bounds[:, 0]) * np.random.random(n_parms) + bounds[:, 0]

        return np.array(best.x)
