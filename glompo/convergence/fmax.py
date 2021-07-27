from .basechecker import BaseChecker

__all__ = ("MaxFuncCalls",)


class MaxFuncCalls(BaseChecker):
    """ Returns :obj:`True` after `fmax` function evaluations have been executed across all managed optimizers. """

    def __init__(self, fmax: int):
        super().__init__()
        self.fmax = fmax

    def __call__(self, manager: 'GloMPOManager') -> bool:
        self.last_result = manager.f_counter >= self.fmax
        return self.last_result
