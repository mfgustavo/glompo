import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import OptimizerLogger

__all__ = ("EvaluationsUnmoving",)


class EvaluationsUnmoving(BaseHunter):

    def __init__(self, calls: int, tol: float = 0):
        """ Returns True if the standard deviation of the last 'calls' function evaluations is below tol * last_f_eval.
            Used to terminate an optimizer when its function evaluations are unchanging, usually indicating that it is
            approaching some convergence. Best used with a hunter which monitors step size to ensure a widely exploring
            optimizer is not killed.
        """
        super().__init__()
        self.calls = calls
        self.tol = tol

    def __call__(self,
                 log: OptimizerLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        vals = log.get_history(victim_opt_id, "fx")
        fcalls = log.get_history(victim_opt_id, "f_call_opt")

        i_crit = np.searchsorted(fcalls - np.max(fcalls) + self.calls, 0)
        # print(f'vals = {vals}')
        # print(f'fcalls = {fcalls}')
        # print(f"icrit = {i_crit}")
        if i_crit == -1:
            # If there are insufficient iterations the hunter will return False
            self._last_result = False
            return self._last_result

        st_dev = np.std(vals[i_crit:])
        # print(f'vals[i_crit:] = {vals[i_crit:]}')
        # print(f'stdev = {st_dev}')
        # print(f'np.abs(vals[-1] * self.tol) = {np.abs(vals[-1] * self.tol)}')
        if np.isnan(st_dev):
            self._last_result = False
            return self._last_result

        self._last_result = st_dev <= np.abs(vals[-1] * self.tol)
        return self._last_result
