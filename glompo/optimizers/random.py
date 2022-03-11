from multiprocessing import Event, Queue
from multiprocessing.connection import Connection
from typing import Callable, Sequence, Tuple

import numpy as np

from .baseoptimizer import BaseOptimizer, MinimizeResult

__all__ = ('RandomOptimizer',)


class RandomOptimizer(BaseOptimizer):
    """ Evaluates random points within the bounds for a fixed number of iterations.
    **Not** actually an optimizer. Intended for debugging.

    Parameters
    ----------
    Inherited, _opt_id _signal_pipe _results_queue _pause_flag _is_log_detailed workers backend
        See :class:`.BaseOptimizer`.
    iters
        Number of function evaluations the optimizer will execute before terminating.
    """

    def __init__(self,
                 _opt_id: int = None,
                 _signal_pipe: Connection = None,
                 _results_queue: Queue = None,
                 _pause_flag: Event = None,
                 _is_log_detailed: bool = False,
                 workers: int = 1,
                 backend: str = 'processes',
                 iters: int = 100):
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, _is_log_detailed, workers, backend)
        self.max_iters = iters
        self.used_iters = 0
        self.result = MinimizeResult()
        self.stop_called = False
        self.logger.debug("Setup optimizer")

    def minimize(self,
                 function: Callable,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:

        best_f = self.result.fx if self.is_restart else np.inf
        bounds = np.transpose(bounds)

        while self.used_iters < self.max_iters:
            if self.stop_called:
                break

            self.used_iters += 1
            vector = np.random.uniform(bounds[0], bounds[1])
            self.logger.debug("Generated vector = %s", vector)

            self.logger.debug("Evaluating function.")
            fx = function(vector)
            self.logger.debug("Function returned fx = %f", fx)

            if callbacks and callbacks():
                self.stop_called = True

            if self._results_queue:
                self.logger.debug("Checking messages")
                self.check_messages()
                self._pause_signal.wait()

            if fx < best_f:
                best_f = fx
                best_x = vector
                self.logger.debug("Updating best")
                self.result.x, self.result.fx = best_x, best_f

        self.result.success = True
        self.logger.debug("Termination successful")
        if self._results_queue:
            self.logger.debug("Messaging manager")
            self.message_manager(0, "Optimizer convergence")
            self.check_messages()

        self.logger.debug("Returning result")
        return self.result

    def callstop(self, reason: str = ""):
        self.stop_called = True
