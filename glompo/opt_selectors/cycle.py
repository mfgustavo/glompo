from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from .baseselector import BaseSelector
from ..core.optimizerlogger import BaseLogger
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("CycleSelector",)


class CycleSelector(BaseSelector):
    """ Iterates and loops through the list of given optimizers each time it is called. """

    def __init__(self,
                 *avail_opts: Union[Type[BaseOptimizer],
                                    Tuple[Type[BaseOptimizer], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
                 allow_spawn: Optional[Callable] = None):
        super().__init__(*avail_opts, allow_spawn=allow_spawn)
        self.i = -1
        self.old = -1

    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: BaseLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None, bool]:

        if not self.allow_spawn(manager):
            return False

        self.old = self.i
        self.i = (self.i + 1) % len(self.avail_opts)
        selected = self.avail_opts[self.i]

        if selected[1]['workers'] > slots_available:
            self.i = self.old
            return None

        return selected
