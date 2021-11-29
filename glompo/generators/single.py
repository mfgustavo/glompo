from typing import Optional, Sequence

import numpy as np

from .basegenerator import BaseGenerator

__all__ = ("SinglePointGenerator",)


class SinglePointGenerator(BaseGenerator):
    """ Always returns the same point.
    Either provided during initialisation or otherwise randomly generated.
    """

    def __init__(self, x: Optional[Sequence[float]] = None):
        super().__init__()
        self.vector = x

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        if self.vector is None:
            bounds = manager.bounds
            n_parms = manager.n_parms
            self.vector = (bounds[:, 1] - bounds[:, 0]) * np.random.random(n_parms) + bounds[:, 0]
        return self.vector.copy()
