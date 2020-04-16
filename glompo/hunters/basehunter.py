

""" Abstract hunter classes used to construct the convergence criteria. """


from abc import ABC, abstractmethod
import inspect

from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("BaseHunter",)


class BaseHunter(ABC):
    """ Base hunter from which all hunters must inherit to be compatible with GloMPO. """

    @abstractmethod
    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        """ When called, this method may check any values within the logs or GPRs of both hunter or the victim
        and return a bool if the desired condition is met.

        Parameters
        ----------
        log: Logger
            Instance of Logger class that contains the iteration history of every optimizer.
        regressor: DataRegressor
            Instance of the DataRegressor class which contains both frequentist and Bayesian methods to regress the data
            against an exponential function.
        hunter_opt_id: int
            ID number of the 'hunter' optimizer currently identified as the best performer.
        victim_opt_id: int
            ID number of the 'victim' optimizer, aspects of which will be compared to the 'hunter' in this class to
            acertain whether it should be shutdown.
        """

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

    def __init__(self, base1: BaseHunter, base2: BaseHunter):
        for base in [base1, base2]:
            if not isinstance(base, BaseHunter):
                raise TypeError("_CombiHunter can only be initialised with instances of BaseHunter subclasses.")
        self.base1 = base1
        self.base2 = base2

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        pass

    def _combi_string_maker(self, keyword: str):
        return f"[{self.base1} {keyword} \n{self.base2}]"


class _AnyHunter(_CombiHunter):
    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        return self.base1.is_kill_condition_met(log, regressor, hunter_opt_id, victim_opt_id) or \
               self.base2.is_kill_condition_met(log, regressor, hunter_opt_id, victim_opt_id)

    def __str__(self):
        return self._combi_string_maker("OR")


class _AllHunter(_CombiHunter):
    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        return self.base1.is_kill_condition_met(log, regressor, hunter_opt_id, victim_opt_id) and \
               self.base2.is_kill_condition_met(log, regressor, hunter_opt_id, victim_opt_id)

    def __str__(self):
        return self._combi_string_maker("AND")
