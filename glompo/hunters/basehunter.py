

""" Abstract hunter classes used to construct the convergence criteria. """


from abc import abstractmethod

from ..core.logger import Logger
from ..core.regression import DataRegressor
from ..common.corebase import *


__all__ = ("BaseHunter",)


class BaseHunter(_CoreBase):
    """ Base hunter from which all hunters must inherit to be compatible with GloMPO. """

    @abstractmethod
    def __call__(self,
                 log: Logger,
                 regressor: DataRegressor,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        """ When called, this method may check any values within the logs or GPRs of both hunter or the victim
            and return a bool if the desired condition is met.

            Note: For proper functionality, the result of __call__ must be saved to self._last_result
                  before returning.

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
                ascertain whether it should be shutdown.
        """
