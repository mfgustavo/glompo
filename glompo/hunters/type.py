from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("TypeHunter",)


class TypeHunter(BaseHunter):
    """ Kills an optimizer based on its class.
    Intended for use with other hunters to allow for specific hunting conditions based on the type of optimizer.

    Parameters
    ----------
    opt_to_kill
        :class:`.BaseOptimizer` class which is targeted.

    Returns
    -------
    bool
        :obj:`True` if the victim is an instance of `opt_to_kill`.

    Examples
    --------
    >>> TypeHunter(CMAOptimizer) & HunterA() | TypeHunter(Nevergrad) & HunterB()

    In this case :code:`HunterA` will only kill :class:`.CMAOptimizer`\\s and :code:`HunterB` will only kill
    :class:`.Nevergrad` optimizers. This is useful in cases where exploratory optimizers should be killed quickly but
    late stage optimizers encouraged to converge and iterate for longer periods.
    """

    def __init__(self, opt_to_kill: BaseOptimizer):
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
