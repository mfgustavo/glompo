import random
from typing import Any, Dict, Tuple, Type, Union

from .baseselector import BaseSelector
from ..core.optimizerlogger import OptimizerLogger
from ..optimizers.baseoptimizer import BaseOptimizer


class RandomSelector(BaseSelector):
    """ Given a set of optimizers, returns a random one when called. """

    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: OptimizerLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None, bool]:

        if not self.allow_spawn(manager):
            return False

        viable = []
        for opt in self.avail_opts:
            if opt[1]['workers'] <= slots_available:
                viable.append(opt)

        if len(viable) == 0:
            return None

        return random.choice(viable)
