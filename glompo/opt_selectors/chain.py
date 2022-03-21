from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .baseselector import BaseSelector
from ..core.optimizerlogger import BaseLogger
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("ChainSelector",)


class ChainSelector(BaseSelector):
    """ Selects the type of optimizer to start based on the number of function evaluations already used.
    Designed to start different types of optimizers at different stages of the optimization. Selects sequentially from
    the list of available optimizers based on the number of function evaluations used.

    Parameters
    ----------
    *avail_opts
        See :class:`.BaseSelector`.
    fcall_thresholds
        A list of length :code:`n` or :code:`n-1`, where :code:`n` is the length of `avail_opts`.

        The first :code:`n-1` elements of this list indicate the function evaluation point at which the selector
        switches to the next type of optimizer in `avail_opts`.

        The optional :code:`n`\\th element indicates the function evaluation at which optimizer spawning is turned off.
    allow_spawn
        See :class:`.BaseSelector`.

    Examples
    --------
    >>> ChainSelector(OptimizerA, OptimizerB, fcall_thresholds=[1000])

    In this case :class:`!OptimizerA` instances will be started in the first 1000 iterations and :class:`!OptimizerB`
    instances will be started thereafter.

    >>> ChainSelector(OptimizerA, OptimizerB, fcall_thresholds=[1000, 2000])

    In this case :class:`!OptimizerA` instances will be started in the first 1000 iterations and :class:`!OptimizerB`
    instances will be started until iteration 2000. No new optimizers will be spawned thereafter.
    """

    def __init__(self,
                 *avail_opts: Union[Type[BaseOptimizer],
                                    Tuple[Type[BaseOptimizer], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
                 fcall_thresholds: List[float],
                 allow_spawn: Optional[List[Callable[['GloMPOManager'], bool]]] = None):
        super().__init__(*avail_opts, allow_spawn=allow_spawn)
        self.fcall_thresholds = fcall_thresholds
        n = len(avail_opts)
        assert n - 1 <= len(fcall_thresholds) <= n, "Must be one threshold less than available optimizers"
        self.toggle = 0

    def select_optimizer(self, manager: 'GloMPOManager', log: BaseLogger, slots_available: int) -> \
            Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]], None, bool]:

        if not all((spawner(manager) for spawner in self.allow_spawn)):
            return False

        if self.toggle < len(self.fcall_thresholds) and manager.f_counter >= self.fcall_thresholds[self.toggle]:
            self.toggle += 1

        selected = self.avail_opts[self.toggle]
        if selected[1]['workers'] > slots_available:
            return None

        return selected
