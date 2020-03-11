

from abc import ABC, abstractmethod
from typing import *


class BaseChecker(ABC):
    """ Base checker from which all checkers must inherit to be compatible with GloMPO. """

    @abstractmethod
    def converged(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has converged and returns a bool. """
        pass

    def __add__(self, other):
        return AnyChecker(self, other)

    def __mul__(self, other):
        return AllChecker(self, other)


class AnyChecker(BaseChecker):
    def __init__(self, base1: BaseChecker, base2: BaseChecker, *args: Sequence[BaseChecker]):
        combi = [base1, base2, *args]
        for base in combi:
            if not isinstance(base, BaseChecker):
                raise TypeError("CombinedChecker can only be initialised with instances of BaseChecker subclasses.")
        self.base_checkers = combi

    def converged(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has converged and returns a bool. """
        return any([base.converged(manager) for base in self.base_checkers])


class AllChecker(BaseChecker):
    def __init__(self, base1: BaseChecker, base2: BaseChecker, *args: Sequence[BaseChecker]):
        combi = [base1, base2, *args]
        for base in combi:
            if not isinstance(base, BaseChecker):
                raise TypeError("CombinedChecker can only be initialised with instances of BaseChecker subclasses.")
        self.base_checkers = combi

    def converged(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has converged and returns a bool. """
        return all([base.converged(manager) for base in self.base_checkers])
