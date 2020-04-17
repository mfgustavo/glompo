

""" Abstract checker classes used to construct the convergence criteria. """


from abc import abstractmethod

from ..common.corebase import _CoreBase


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
