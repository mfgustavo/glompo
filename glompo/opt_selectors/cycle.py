

from typing import *

from .baseselector import BaseSelector
from ..core.optimizerlogger import OptimizerLogger
from ..optimizers.baseoptimizer import BaseOptimizer


__all__ = ("CycleSelector",)


class CycleSelector(BaseSelector):
    """ Iterates and loops through the list of given optimizers each time it is called. """

    def __init__(self,
                 avail_opts: List[Union[Type[BaseOptimizer],
                                        Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]]):
        super().__init__(avail_opts)
        self.i = -1

    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: OptimizerLogger) -> Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]:
        self.i = self.i + 1 if self.i < len(self.avail_opts) - 1 else 0
        return self.avail_opts[self.i]
