
from typing import *
import logging
from multiprocessing import Event, Queue
from multiprocessing.connection import Connection

import numpy as np

from .baseoptimizer import MinimizeResult, BaseOptimizer
from ..common.namedtuples import IterationResult


__all__ = ('RandomOptimizer',)


class RandomOptimizer(BaseOptimizer):
    """
    Evaluates random points within the bounds for a fixed number of iterations (used for debugging).
    """

    def __init__(self, opt_id: int = None, signal_pipe: Connection = None, results_queue: Queue = None,
                 pause_flag: Event = None, workers: int = 1, iters: int = 100):
        """ Initialize with the above parameters. """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers)
        self.logger = logging.getLogger('glompo.optimizers')
        self.iters = iters
        self.result = None
        self.stop_called = False
        self.logger.debug("Setup optimizer")

    def minimize(self,
                 function: Callable,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:

        vector = []
        fx = np.inf
        best_f = np.inf
        for i in range(self.iters):
            if self.stop_called:
                break

            vector = []
            for bnd in bounds:
                vector.append(np.random.uniform(bnd[0], bnd[1]))
            vector = np.array(vector)
            self.logger.debug(f"Generated vector = {vector}")

            self.logger.debug(f"Evaluating function.")
            fx = function(vector)
            self.logger.debug(f"Function returned fx = {fx}")

            if self._results_queue:
                self.logger.debug(f"Pushing result")
                self.push_iter_result(i+1, 1, vector, fx, False)
                self.logger.debug(f"Checking messages")
                self.check_messages()
                self._pause_signal.wait()

            if fx < best_f:
                best_f = fx
                best_x = vector
                self.logger.debug(f"Updating best")
                self.result = (best_x, best_f)

        self.result.success = True
        self.logger.debug("Termination successful")
        if self._results_queue:
            self.push_iter_result(self.iters, 1, vector, fx, True)
            self.logger.debug("Messaging manager")
            self.message_manager(0, "Optimizer convergence")
            self.check_messages()

        self.logger.debug("Returning result")
        return self.result

    def push_iter_result(self, i, f_calls, x, fx, last):
        self._results_queue.put(IterationResult(self._opt_id, i, f_calls, x, fx, last))

    def callstop(self, reason=None):
        self.stop_called = True

    def save_state(self, *args):
        pass
