from .basechecker import BaseChecker

__all__ = ("NOptConverged",)


class NOptConverged(BaseChecker):
    """ Returns :obj:`True` when `nconv` optimizers have converged normally.
    'Normally' here is defined as exiting the minimization loop according to the optimizer's own internal convergence
    criteria, rather than any GloMPO intervention.
    """

    def __init__(self, nconv: int):
        super().__init__()
        self.nconv = nconv

    def __call__(self, manager: 'GloMPOManager') -> bool:
        self.last_result = manager.conv_counter >= self.nconv
        return self.last_result
