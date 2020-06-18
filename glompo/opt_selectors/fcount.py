

from typing import *

from ..core.optimizerlogger import OptimizerLogger
from ..optimizers.baseoptimizer import BaseOptimizer
from .baseselector import BaseSelector


__all__ = ("FCallsSelector",)


class FCallsSelector(BaseSelector):
    """ Selects the type of generator to start based on the number of function evaluations already used. Optimizers
        better at global search should be used early and optimizers better at local optimization should be used later.
    """

    def __init__(self,
                 avail_opts: List[Union[Type[BaseOptimizer],
                                        Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]],
                 fcall_thresholds: List[float]):
        """

        Parameters
        ----------
        avail_opts: List[Union[Type[BaseOptimizer], Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]]
            List of available optimizers and their configurations in order of usage.
        fcall_thresholds: List[float]
            A list of length n-1 where n is the length of avail_opts. Indicates the iteration number at which the
            selector will switch to selecting the next optimizer in the list.
        """
        super().__init__(avail_opts)
        self.avail_opts = avail_opts
        self.fcall_thresholds = fcall_thresholds
        self.toggle = 0

    def select_optimizer(self, manager: 'GloMPOManager', log: OptimizerLogger) -> \
            Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]:

        if self.toggle < len(self.fcall_thresholds) and manager.f_counter > self.fcall_thresholds[self.toggle]:
            self.toggle += 1

        return self.avail_opts[self.toggle]
