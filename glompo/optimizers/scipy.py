import warnings
from multiprocessing import Event
from multiprocessing.connection import Connection
from queue import Queue
from typing import Callable, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from .baseoptimizer import BaseOptimizer, MinimizeResult

__all__ = ("ScipyOptimizerWrapper",)

warnings.filterwarnings('error', module="glompo.optimizers.scipy")


class GloMPOCallstop(Exception):
    """ Custom Exception used by GloMPO to stop Scipy optimizers early. """


class ScipyOptimizerWrapper(BaseOptimizer):
    """ Very rough wrapper around scipy optimizers.
    However, the code is very impregnable so only the simplest GloMPO functionality is supported (i.e. early termination
    from the manager). Other things like checkpointing are not supported.

    Also take note of the various scipy limitations of which methods can be combined with bounds. This class
    automatically raises warnings to catch these incompatibilities.
    """

    def __init__(self,
                 _opt_id: int = None,
                 _signal_pipe: Connection = None,
                 _results_queue: Queue = None,
                 _pause_flag: Event = None,
                 workers: int = 1,
                 backend: str = 'processes',
                 is_log_detailed: bool = False,
                 method: str = 'Nelder-Mead'):
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, workers, backend, is_log_detailed)
        self.stop = False
        self.opt_method = method

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        try:
            sp_result = minimize(fun=function,
                                 x0=np.array(x0),
                                 method=self.opt_method,
                                 bounds=bounds,
                                 callback=_GloMPOCallbacksWrapper(self, callbacks), **kwargs)
        except GloMPOCallstop:
            return

        if self._results_queue:
            self.message_manager(0, "Optimizer convergence")

        result = MinimizeResult()
        result.success = sp_result.success
        result.x = sp_result.x
        result.fx = sp_result.fun

        return result

    def callstop(self, *args):
        self.stop = True


class _GloMPOCallbacksWrapper:
    """ Wraps all the components needed by GloMPO to be called after each iteration into a single object which can be
        registered as a callback.
    """

    def __init__(self,
                 parent: BaseOptimizer,
                 callbacks: Union[None,
                                  Callable[..., bool],
                                  Sequence[Callable[..., bool]]] = None):
        self.parent = parent
        self.n_calls = 0

        if callable(callbacks):
            self.callbacks = [callbacks]
        elif callbacks is None:
            self.callbacks = []

    def __call__(self, *args, **kwargs):
        self.n_calls += 1

        stop_cond = None

        # User sent callbacks
        if any([cb(*args, **kwargs) for cb in self.callbacks]):
            stop_cond = "Direct user callbacks"
        self.parent.logger.debug("Stop = %s at user callbacks (iter: %d)", bool(stop_cond), self.n_calls)

        # GloMPO specific callbacks
        if self.parent._results_queue:
            self.parent._pause_signal.wait()
            self.parent.check_messages()
            if not stop_cond and self.parent.stop:
                stop_cond = "GloMPO termination signal."
            self.parent.logger.debug("Stop = %s after message check from manager", bool(stop_cond))

        if stop_cond:
            self.parent.logger.debug("Stop is True so shutting down optimizer.")
            self.parent.message_manager(0, stop_cond)
            raise GloMPOCallstop
