import logging
from abc import abstractmethod

from ..common.corebase import _AndCore, _CoreBase, _OrCore

__all__ = ("BaseChecker",)


class BaseChecker(_CoreBase):
    """ Abstract class from which all checkers must inherit to be compatible with GloMPO.

    Attributes
    ----------
    logger : logging.Logger
        :class:`logging.Logger` instance into which status messages may be added.
    """

    def __init__(self):
        self.logger = logging.getLogger('glompo.checker')
        super().__init__()

    @abstractmethod
    def __call__(self, manager: 'GloMPOManager') -> bool:
        """ Evaluates if the checker condition is met.

        Parameters
        ----------
        manager
            :class:`.GloMPOManager` instance which is managing the optimization. Its attributes can be accessed when
            determining the convergence criteria.

        Returns
        -------
        bool
            :obj:`True` if the convergence criteria is met, :obj:`False` otherwise.

        Notes
        -----
        For proper functionality, the result of this method must be saved to :attr:`last_result` before
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
