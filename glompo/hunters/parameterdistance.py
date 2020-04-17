
import numpy as np

from .basehunter import BaseHunter
from ..core.logger import Logger
from ..core.regression import DataRegressor


__all__ = ("ParameterDistance",)


class ParameterDistance(BaseHunter):
    """ Calculates the Euclidean distance in parameter space between the first and last optimizer iterations
        for the hunter (trajectory length). Calculates the Euclidean distance between the final
        iterations of the hunter and victim (inter-optimizer distance).

        If the fraction between the inter-optimizer distance and trajectory length is less than the provided
        tolerance the optimizers are deemed to be near one another and the hunter returns True.
    """

    def __init__(self, relative_distance: float):
        super().__init__()
        if isinstance(relative_distance, (float, int)) and relative_distance > 0:
            self.relative_distance = relative_distance
        else:
            raise ValueError("relative_distance should be a positive float.")

    def is_kill_condition_met(self,
                              log: Logger,
                              regressor: DataRegressor,
                              hunter_opt_id: int,
                              victim_opt_id: int) -> bool:
        h0 = np.array(log.get_history(hunter_opt_id, 'x')[0])
        h1 = np.array(log.get_history(hunter_opt_id, 'x')[-1])
        v1 = np.array(log.get_history(victim_opt_id, 'x')[-1])

        traj_length = np.sqrt(np.sum((h0 - h1) ** 2))
        opt_dist = np.sqrt(np.sum((v1 - h1) ** 2))
        ratio = opt_dist / traj_length

        self._kill_result = ratio <= self.relative_distance
        return self._kill_result
