

from typing import *

from ..core.optimizerlogger import OptimizerLogger
from ..optimizers.baseoptimizer import BaseOptimizer
from .baseselector import BaseSelector


__all__ = ("ChainSelector",)


class ChainSelector(BaseSelector):
    """ Selects the type of generator to start based on the number of function evaluations already used.
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
        self.fcall_thresholds = fcall_thresholds
        assert len(avail_opts) == len(fcall_thresholds) + 1, "Must be one threshold less than available optimizers"
        self.toggle = 0

    def select_optimizer(self, manager: 'GloMPOManager', log: OptimizerLogger, slots_available: int) -> \
            Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]], None]:

        if self.toggle < len(self.fcall_thresholds) and manager.f_counter >= self.fcall_thresholds[self.toggle]:
            self.toggle += 1

        selected = self.avail_opts[self.toggle]
        if selected[1]['workers'] > slots_available:
            return None

        return selected
