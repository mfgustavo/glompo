

""" Base generator from which all generators must inherit to be compatible with GloMPO. """

from abc import ABC, abstractmethod


class BaseChecker(ABC):
    @abstractmethod
    def converged(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has converged and returns a bool. """
        pass
