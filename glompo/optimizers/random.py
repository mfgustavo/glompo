from multiprocessing import Event, Queue
from multiprocessing.connection import Connection
from typing import Callable, Sequence, Tuple

import numpy as np

from .baseoptimizer import BaseOptimizer, MinimizeResult

__all__ = ('RandomOptimizer',)


class RandomOptimizer(BaseOptimizer):
    """
    Evaluates random points within the bounds for a fixed number of iterations (used for debugging).
    """

    def __init__(self,
                 opt_id: int = None,
                 signal_pipe: Connection = None,
                 results_queue: Queue = None,
                 pause_flag: Event = None,
                 workers: int = 1,
                 backend: str = 'processes',
                 is_log_detailed: bool = False,
                 iters: int = 100):
        """ Initialize with the above parameters. """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag,
                         workers, backend, is_log_detailed)
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

        while self.used_iters < self.max_iters:
            if self.stop_called:
                break

            self.used_iters += 1
            vector = []
            for bnd in bounds:
                vector.append(np.random.uniform(bnd[0], bnd[1]))
            vector = np.array(vector)
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
