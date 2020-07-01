

from typing import List, Union, Type, Tuple, Dict, Any

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
        self.old = -1

    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: OptimizerLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None]:
        self.old = self.i
        self.i = (self.i + 1) % len(self.avail_opts)
        selected = self.avail_opts[self.i]

        if selected[1]['workers'] > slots_available:
            self.i = self.old
            return None

        return selected
