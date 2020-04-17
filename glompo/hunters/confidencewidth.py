
from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("ConfidenceWidth",)


class ConfidenceWidth(BaseHunter):

    def __init__(self, threshold: float):
        """ Returns True if the standard deviation of the victim's asymptote uncertainty is less than a percentage of
            the mean. The fraction is given by threshold as a value between 0 and 1.
        """
        super().__init__()
        if isinstance(threshold, (float, int)) and threshold > 0:
            self.threshold = threshold
        else:
            raise ValueError("threshold should be a positive float.")

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        med, lower, upper = regressor.get_mcmc_results(victim_opt_id, 'Asymptote')

        self._kill_result = (upper - lower) < self.threshold * abs(med)
        return self._kill_result
