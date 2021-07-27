from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("TypeHunter",)


class TypeHunter(BaseHunter):

    def __init__(self, opt_to_kill: BaseOptimizer):
        """ Returns True if the victim is of a certain optimizer type. Intended use is in combination with other
            hunters to allow type specific hunting.

            For example: TypeHunter(CMAOptimizer) & HunterA() | TypeHunter(Nevergrad) & HunterB()
                In this case HunterA will only kill CMAOptimizers and HunterB will only kill Nevergrad optimizers.
                This is useful in cases where exploratory optimizers should be killed quickly but late stage
                optimizers encouraged to converge and iterate for longer periods
        """
        super().__init__()
        if issubclass(opt_to_kill, BaseOptimizer):
            self.opt_to_kill = opt_to_kill.__name__
        else:
            raise TypeError("Optimizer not recognized, must be a subclass of BaseOptimizer")

    def __call__(self,
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:

        self.last_result = self.opt_to_kill == log.get_metadata(victim_opt_id, "opt_type")
        return self.last_result
