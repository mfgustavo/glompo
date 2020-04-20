

""" Abstract checker classes used to construct the convergence criteria. """


from abc import abstractmethod

from ..common.corebase import _CoreBase, _OrCore, _AndCore


__all__ = ("BaseChecker",)


class BaseChecker(_CoreBase):
    """ Base checker from which all checkers must inherit to be compatible with GloMPO. """

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


class _OrChecker(BaseChecker, _OrCore):
    def __call__(self, manager: 'GloMPOManager') -> bool:
        return super(BaseChecker, self).__call__(manager)


class _AndChecker(BaseChecker, _AndCore):
    def __call__(self, manager: 'GloMPOManager') -> bool:
        return super(BaseChecker, self).__call__(manager)

