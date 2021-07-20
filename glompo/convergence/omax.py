from .basechecker import BaseChecker

__all__ = ("MaxOptsStarted",)


class MaxOptsStarted(BaseChecker):
    """ Returns :obj:`True` after `omax` optimizers have been started. """

    def __init__(self, omax: int):
        super().__init__()
        self.omax = omax

    def __call__(self, manager: 'GloMPOManager') -> bool:
        self.last_result = manager.o_counter >= self.omax
        return self.last_result
