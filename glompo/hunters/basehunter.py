""" Abstract hunter classes used to construct the convergence criteria. """

import logging
from abc import abstractmethod

from ..common.corebase import _AndCore, _CoreBase, _OrCore
from ..core.optimizerlogger import OptimizerLogger

__all__ = ("BaseHunter",)


class BaseHunter(_CoreBase):
    """ Base hunter from which all hunters must inherit to be compatible with GloMPO. """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('glompo.hunter')

    @abstractmethod
    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        """ When called, this method may check any values within the logs or GPRs of both hunter or the victim
            and return a bool if the desired condition is met.

            Note: For proper functionality, the result of __call__ must be saved to self._last_result
                  before returning.

            Parameters
            ----------
            log: OptimizerLogger
                Instance of Logger class that contains the iteration history of every optimizer.
            hunter_opt_id: int
                ID number of the 'hunter' optimizer currently identified as the best performer.
            victim_opt_id: int
                ID number of the 'victim' optimizer, aspects of which will be compared to the 'hunter' in this class to
                ascertain whether it should be shutdown.
        """

    def __or__(self, other: 'BaseHunter') -> '_OrHunter':
        return _OrHunter(self, other)

    def __and__(self, other: 'BaseHunter') -> '_AndHunter':
        return _AndHunter(self, other)


class _OrHunter(_OrCore, BaseHunter):
    def __call__(self, *args, **kwargs) -> bool:
        return super().__call__(*args, **kwargs)


class _AndHunter(_AndCore, BaseHunter):
    def __call__(self, *args, **kwargs) -> bool:
        return super().__call__(*args, **kwargs)
