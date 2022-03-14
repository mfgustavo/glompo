from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.connection import Connection
from queue import Queue
from threading import Event
from typing import Callable, Optional, Sequence, Tuple

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
                 iters: int = 100,
                 seed: Optional[int] = None):
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, _is_log_detailed, workers, backend)
        self.max_iters = iters
        self.used_iters = 0
        self.result = MinimizeResult()
        self.stop_called = False
        self.seed = seed
        self.logger.debug("Setup optimizer")

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 **kwargs) -> MinimizeResult:
        if self.seed:
            np.random.seed(self.seed)

        bounds = np.transpose(bounds)

        def evaluate(_):
            x = np.random.uniform(bounds[0], bounds[1])
            fx = function(x)
            return x, fx

        if self.workers > 1:
            if self._backend == 'threads':
                executor = ThreadPoolExecutor(self.workers)
            else:
                executor = ProcessPoolExecutor(self.workers)
            generator = executor.map(evaluate, range(self.used_iters, self.max_iters))
        else:
            generator = map(evaluate, range(self.used_iters, self.max_iters))

        for x, fx in generator:
            self.used_iters += 1  # Used for checkpointing and restarts
            if fx < self.result.fx:
                self.result.fx = fx
                self.result.x = x
            if self._results_queue:
                self.check_messages()
                self._pause_signal.wait()
            if self.stop_called:
                break

        self.result.success = not bool(self.stop_called)

        if self.workers > 1:
            executor.shutdown()

        if self._results_queue:
            self.logger.debug("Messaging manager")
            self.message_manager(0, "Optimizer convergence")
            self.check_messages()

        return self.result

    def callstop(self, reason: str = ""):
        self.stop_called = True
