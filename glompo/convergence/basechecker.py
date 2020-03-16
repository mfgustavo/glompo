

from abc import ABC, abstractmethod
from typing import *


class BaseChecker(ABC):
    """ Base checker from which all checkers must inherit to be compatible with GloMPO. """

    def __init__(self):
        self._converged = False
        
    @abstractmethod
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has _converged and returns a bool.

            Note: For proper functionality, the result of check_convergence must be saved to self._converged before
            returning. """
        pass

    def __add__(self, other):
        return _AnyChecker(self, other)

    def __mul__(self, other):
        return _AllChecker(self, other)
    
    def __str__(self) -> str:
        vars = ""
        attrs = [item for item in dir(self) if
                 not item.startswith("_") and
                 not callable(self.__getattribute__(item)) and
                 item != "_converged"]
        for item in attrs:
            if item != "check_convergence":
                vars += f"{item}={self.__getattribute__(item)}, "
        vars = vars[:-2]
        return f"{self.__class__.__name__}({vars})"


class _CombiChecker(BaseChecker):

    def __init__(self, base1: BaseChecker, base2: BaseChecker, *args: Sequence[BaseChecker]):
        super().__init__()
        combi = [base1, base2, *args]
        for base in combi:
            if not isinstance(base, BaseChecker):
                raise TypeError("_CombiChecker can only be initialised with instances of BaseChecker subclasses.")
        self.base_checkers = combi
        self.converged = False

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        pass


class _AnyChecker(_CombiChecker):
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self.converged = any([base.check_convergence(manager) for base in self.base_checkers])
        return self.converged

    def __str__(self):
        mess = ""
        for base in self.base_checkers:
            mess += f"{base} OR \n"
        mess = mess[:-5]
        return mess


class _AllChecker(_CombiChecker):
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self.converged = all([base.check_convergence(manager) for base in self.base_checkers])
        return self.converged

    def __str__(self):
        mess = ""
        for base in self.base_checkers:
            mess += f"{base} AND \n"
        mess = mess[:-5]
        return mess
