from .basechecker import BaseChecker

__all__ = ("TargetCost",)


class TargetCost(BaseChecker):
    """ Returns `f_best <= target + atol`, where `f_best` is the best value seen thus far by the manager. """

    def __init__(self, target: float, atol: float = 1E-6):
        super().__init__()
        self.target = target
        self.atol = atol

    def __call__(self, manager: 'GloMPOManager') -> bool:
        if manager.result.fx is not None:
            self.last_result = manager.result.fx <= self.target + self.atol
        else:
            self.last_result = False

        return self.last_result
