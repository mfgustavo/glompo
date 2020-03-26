

""" Abstract hunter classes used to construct the convergence criteria. """


from abc import ABC, abstractmethod
from typing import *
import inspect

from ..core.logger import Logger
from ..core.gpr import GaussianProcessRegression


__all__ = ("BaseHunter",)


class BaseHunter(ABC):
    """ Base hunter from which all hunters must inherit to be compatible with GloMPO. """

    @abstractmethod
    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        """ When called, this method may check any values within the logs or GPRs of both hunter or the victim
        and return a bool if the desired condition is met. """

    def __or__(self, other: 'BaseHunter') -> '_AnyHunter':
        return _AnyHunter(self, other)

    def __and__(self, other: 'BaseHunter') -> '_AllHunter':
        return _AllHunter(self, other)

    def __str__(self) -> str:
        lst = ""
        signature = inspect.signature(self.__init__)
        for parm in signature.parameters:
            if parm in dir(self):
                lst += f"{parm}={self.__getattribute__(parm)}, "
            else:
                lst += f"{parm}, "
        lst = lst[:-2]
        return f"{self.__class__.__name__}({lst})"


class _CombiHunter(BaseHunter):

    def __init__(self, base1: BaseHunter, base2: BaseHunter, *args: Sequence[BaseHunter]):
        combi = [base1, base2, *args]
        for base in combi:
            if not isinstance(base, BaseHunter):
                raise TypeError("_CombiHunter can only be initialised with instances of BaseChecker subclasses.")
        self.base_checkers = combi

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        pass


class _AnyHunter(_CombiHunter):
    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        return any([base.is_kill_condition_met(log, hunter_opt_id, hunter_gpr, victim_opt_id, victim_gpr) for base in
                    self.base_checkers])

    def __str__(self):
        mess = ""
        for base in self.base_checkers:
            mess += f"{base} OR \n"
        mess = mess[:-5]
        return mess


class _AllHunter(_CombiHunter):
    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        return all([base.is_kill_condition_met(log, hunter_opt_id, hunter_gpr, victim_opt_id, victim_gpr) for base in
                    self.base_checkers])

    def __str__(self):
        mess = ""
        for base in self.base_checkers:
            mess += f"{base} AND \n"
        mess = mess[:-6]
        return mess
