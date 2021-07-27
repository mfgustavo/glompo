""" Abstract hunter classes used to construct the convergence criteria. """

import logging
from abc import abstractmethod

from ..common.corebase import _AndCore, _CoreBase, _OrCore
from ..core.optimizerlogger import BaseLogger

__all__ = ("BaseHunter",)


class BaseHunter(_CoreBase):
    """ Base hunter from which all hunters must inherit to be compatible with GloMPO. """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('glompo.hunter')

    @abstractmethod
    def __call__(self,
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        """ Compares two optimizers and returns if one should be terminated according to some condition.
        When called, this method may check any values within the logs of both the hunter or the victim.

        Parameters
        ----------
        log
            Instance of :class:`.BaseLogger` that contains the iteration history of every optimizer.
        hunter_opt_id
            ID number of the 'hunter'; defined as the optimizer which has found the lowest function value.
        victim_opt_id
            ID number of the 'victim'; defined as any optimizer which is not the hunter.

        Returns
        -------
        bool
            :obj:`True` if the condition in the method is met, :obj:`False` otherwise.

        Notes
        -----
        For proper functionality, the result of this method must be saved to :attr:`last_result` before
        returning.
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
