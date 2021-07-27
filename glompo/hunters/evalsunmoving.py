import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger

__all__ = ("EvaluationsUnmoving",)


class EvaluationsUnmoving(BaseHunter):
    """ Considers function values the optimizers are currently exploring.
    Used to terminate an optimizer when its function evaluations are unchanging, usually indicating that it is
    approaching some convergence. Best used with a hunter which monitors step size to ensure a widely exploring
    optimizer is not killed.

    Parameters
    ----------
    calls
        Number of function evaluations between comparison points.
    tol
        Tolerance fraction between 0 and 1.

    Returns
    -------
    bool
        Returns :obj:`True` if the standard deviation of the last `calls` function evaluations is below
        :code:`tol * abs(latest_f_eval)`.
    """

    def __init__(self, calls: int, tol: float = 0):
        super().__init__()
        self.calls = calls
        self.tol = tol

    def __call__(self,
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        vals = log.get_history(victim_opt_id, "fx")
        n_calls = log.len(victim_opt_id)

        if n_calls < self.calls:
            # If there are insufficient iterations the hunter will return False
            self.last_result = False
            return self.last_result

        try:
            st_dev = np.std(vals[-self.calls:])
        except FloatingPointError:
            vals = np.array(vals)[np.isfinite(vals)]
            st_dev = np.std(vals[-self.calls:])

        self.last_result = st_dev <= np.abs(vals[-1] * self.tol)
        return self.last_result
