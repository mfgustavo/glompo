from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .baseselector import BaseSelector
from ..core.optimizerlogger import OptimizerLogger
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("ChainSelector",)


class ChainSelector(BaseSelector):
    """ Selects the type of generator to start based on the number of function evaluations already used.
    """

    def __init__(self,
                 avail_opts: List[Union[Type[BaseOptimizer],
                                        Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]],
                 fcall_thresholds: List[float],
                 allow_spawn: Optional[Callable[['GloMPOManager'], bool]] = None):
        """
        Parameters
        ----------
        avail_opts: List[Union[Type[BaseOptimizer], Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]]
            List of available optimizers and their configurations in order of usage.
        fcall_thresholds: List[float]
            A list with length equal to n or n-1 where n is the length of avail_opts.

            The first n-1 elements of this list indicate the function evaluation point at which the selector switches to
            the next type of optimizer in avail_opts.

            The optional nth element indicates the function evaluation at which optimizer spawning is turned off.
        allow_spawn: Optional[Callable[['GloMPOManager'], bool]]
                Optional function sent to the selector which is called with the manager object as argument. If it
                returns False the manager will no longer spawn optimizers.

        Examples
        --------
        ChainSelector([OptimizerA, OptimizerB], [1000])
            In this case OptimizerA instances will be started in the first 1000 iterations and OptimizerB instances will
            be started thereafter.
        ChainSelector([OptimizerA, OptimizerB], [1000, 2000])
            In this case OptimizerA instances will be started in the first 1000 iterations and OptimizerB instances will
            be started until iteration 2000. No new optimizers will be spawned thereafter.
        """
        super().__init__(avail_opts, allow_spawn)
        self.fcall_thresholds = fcall_thresholds
        n = len(avail_opts)
        assert n - 1 <= len(fcall_thresholds) <= n, "Must be one threshold less than available optimizers"
        self.toggle = 0

    def select_optimizer(self, manager: 'GloMPOManager', log: OptimizerLogger, slots_available: int) -> \
            Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]], None, bool]:

        if not self.allow_spawn(manager):
            return False

        if self.toggle < len(self.fcall_thresholds) and manager.f_counter >= self.fcall_thresholds[self.toggle]:
            self.toggle += 1

        try:
            selected = self.avail_opts[self.toggle]
            if selected[1]['workers'] > slots_available:
                return None
        except IndexError:
            return False

        return selected
