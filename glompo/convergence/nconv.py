

from .basechecker import BaseChecker


__all__ = ("NOptConverged",)


class NOptConverged(BaseChecker):

    def __init__(self, nconv: int):
        """ Convergence is reached after nconv optimizers have been converged normally. """
        super().__init__()
        self.nconv = nconv

    def __call__(self, manager: 'GloMPOManager') -> bool:
        self._last_result = manager.conv_counter >= self.nconv
        return self._last_result
