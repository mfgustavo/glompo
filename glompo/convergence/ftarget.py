import numpy as np

from .basechecker import BaseChecker

__all__ = ("TargetCost",)


class TargetCost(BaseChecker):

    def __init__(self, target: float, atol: float = 1E-6):
        """ Convergence is reached when the |f_best - target| < atol where f_best is the best value seen thus far
            by the optimizer.
        """
        super().__init__()
        self.target = target
        self.atol = atol

    def __call__(self, manager: 'GloMPOManager') -> bool:
        if manager.result.fx is not None:
            self._last_result = np.abs(manager.result.fx - self.target) < self.atol
        else:
            self._last_result = False

        return self._last_result
