

""" Abstract checker classes used to construct the convergence criteria. """


import logging
from abc import abstractmethod

from ..common.corebase import _CoreBase, _OrCore, _AndCore


__all__ = ("BaseChecker",)


class BaseChecker(_CoreBase):
    """ Base checker from which all checkers must inherit to be compatible with GloMPO. """

    def __init__(self):
        self.logger = logging.getLogger('glompo.checker')
        super().__init__()

    @abstractmethod
    def __call__(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has converged and returns a bool.

            Note: For proper functionality, the result of __call__ must be saved to self._last_result before
            returning.
        """

    def __or__(self, other: 'BaseChecker') -> '_OrChecker':
        return _OrChecker(self, other)

    def __and__(self, other: 'BaseChecker') -> '_AndChecker':
        return _AndChecker(self, other)


class _OrChecker(_OrCore, BaseChecker):
    def __call__(self, manager: 'GloMPOManager') -> bool:
        return super().__call__(manager)


class _AndChecker(_AndCore, BaseChecker):
    def __call__(self, manager: 'GloMPOManager') -> bool:
        return super().__call__(manager)

