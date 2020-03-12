

from abc import ABC, abstractmethod
from typing import *


class BaseChecker(ABC):
    """ Base checker from which all checkers must inherit to be compatible with GloMPO. """

    def __init__(self):
        self.converged = False
        
    @abstractmethod
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has converged and returns a bool.

            Note: For proper functionality, the result of check_convergence must be saved to self.converged before
            returning. """
        pass

    def __add__(self, other):
        return _AnyChecker(self, other)

    def __mul__(self, other):
        return _AllChecker(self, other)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.converged}"


class _CombiChecker(BaseChecker):

    def __init__(self, base1: BaseChecker, base2: BaseChecker, *args: Sequence[BaseChecker]):
        super().__init__()
        combi = [base1, base2, *args]
        for base in combi:
            if not isinstance(base, BaseChecker):
                raise TypeError("CombinedChecker can only be initialised with instances of BaseChecker subclasses.")
        self.base_checkers = combi
        self.converged = False

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        pass

    def __str__(self):
        mess = ""
        for base in self.base_checkers:
            mess += f"{base}\n"
        mess = mess[:-1]
        return mess


class _AnyChecker(_CombiChecker):
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self.converged = any([base.check_convergence(manager) for base in self.base_checkers])
        return self.converged


class _AllChecker(_CombiChecker):
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self.converged = all([base.check_convergence(manager) for base in self.base_checkers])
        return self.converged
