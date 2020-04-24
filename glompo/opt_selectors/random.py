

from typing import *
import random

from .baseselector import BaseSelector
from ..core.optimizerlogger import OptimizerLogger
from ..optimizers.baseoptimizer import BaseOptimizer


class RandomSelector(BaseSelector):
    """ Given a set of optimizers, returns a random one when called. """

    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: OptimizerLogger) -> Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]:
        return random.choice(self.avail_opts)
