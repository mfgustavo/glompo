
from abc import ABC, abstractmethod
from typing import *

from ..core.logger import Logger
from ..core.gpr import GaussianProcessRegression


class BaseHunter(ABC):
    """ Base hunter from which all hunters must inherit to be compatible with GloMPO. """

    @abstractmethod
    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        """ When called, this method may check any values within the logs or GPRs of both hunter or the victim
        and return a bool if the desired condition is met. """
        pass

    def __add__(self, other):
        return _AnyHunter(self, other)

    def __mul__(self, other):
        return _AllHunter(self, other)


class _CombiChecker(BaseHunter):

    def __init__(self, base1: BaseHunter, base2: BaseHunter, *args: Sequence[BaseHunter]):
        combi = [base1, base2, *args]
        for base in combi:
            if not isinstance(base, BaseHunter):
                raise TypeError("_CombiHunter can only be initialised with instances of BaseChecker subclasses.")
        self.base_checkers = combi

    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        pass


class _AnyHunter(_CombiChecker):
    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        return any([base.is_kill_condition_met(log, hunter_opt_id, hunter_gpr, victim_opt_id, victim_gpr) for base in
                    self.base_checkers])


class _AllHunter(_CombiChecker):
    def is_kill_condition_met(self, log: Logger, hunter_opt_id: int, hunter_gpr: GaussianProcessRegression,
                              victim_opt_id: int, victim_gpr: GaussianProcessRegression) -> bool:
        return all([base.is_kill_condition_met(log, hunter_opt_id, hunter_gpr, victim_opt_id, victim_gpr) for base in
                    self.base_checkers])
