from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice, product
from multiprocessing.connection import Connection
from threading import Event

import numpy as np
from typing import Callable, Optional, Sequence, Tuple

from .baseoptimizer import BaseOptimizer, MinimizeResult
from ..core._backends import ChunkingQueue

__all__ = ('SimpleGridOptimizer',)


class SimpleGridOptimizer(BaseOptimizer):
    """ A simple example optimizer, that just evaluates the fitness function on a regular grid in parameter space. """

    def __init__(self,
                 _opt_id: Optional[int] = None,
                 _signal_pipe: Optional[Connection] = None,
                 _results_queue: Optional[ChunkingQueue] = None,
                 _pause_flag: Optional[Event] = None,
                 _is_log_detailed: bool = False,
                 workers: int = 1,
                 backend: str = 'threads',
                 nsteps=10):
        """Create a new optimizer for a regular grid of `nsteps` points in every active parameter's range."""
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, _is_log_detailed, workers, backend,
                         nsteps=nsteps)
        self.nsteps = nsteps
        self.stop = False
        self.n_processed = 0
        self.result = MinimizeResult()

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 **kwargs) -> MinimizeResult:
        """ Evaluate loss function for all grid points and return an instance of :class:`MinimizeResult` for the best
            grid point.
        """
        if len(bounds) != len(x0):
            raise ValueError('Size mismatch between x0 and bounds')

        bounds = np.transpose(bounds)
        pargrid = np.linspace(bounds[0], bounds[1], self.nsteps + 1).T

        def full_return(x):
            if self.stop:
                raise StopIteration
            return x, function(x)

        if self.workers > 1:
            if self._backend == 'threads':
                executor = ThreadPoolExecutor(self.workers)
            else:
                executor = ProcessPoolExecutor(self.workers)
            generator = executor.map(full_return, islice(product(*pargrid), self.n_processed, None))
        else:
            generator = map(full_return, islice(product(*pargrid), self.n_processed, None))

        for x, fx in generator:
            self.n_processed += 1  # Used for checkpointing and restarts
            if fx < self.result.fx:
                self.result.fx = fx
                self.result.x = x
            if self._results_queue:
                self.check_messages()
                self._pause_signal.wait()
            if self.stop:
                break

        self.result.success = not bool(self.stop)

        if self.workers > 1:
            executor.shutdown()

        if self._results_queue:
            self.logger.debug("Messaging manager")
            self.message_manager(0, "Optimizer convergence")
            self.check_messages()

        return self.result

    def callstop(self, reason: str = ""):
        self.stop = True
