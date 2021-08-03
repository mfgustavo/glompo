import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Event, Queue
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Callable, Optional, Sequence, Set, Union

import nevergrad as ng
import numpy as np

from .baseoptimizer import BaseOptimizer, MinimizeResult

__all__ = ('Nevergrad',)


class Nevergrad(BaseOptimizer):
    """ Provides access to the optimizers available through the
    `nevergrad <https://facebookresearch.github.io/nevergrad/>`_ package.

    Parameters
    ----------
    Inherited, _opt_id _signal_pipe _results_queue _pause_flag workers backend is_log_detailed
        See :class:`.BaseOptimizer`.
    optimizer
        String key to the desired optimizer. See nevergrad documentation for a list of available algorithms.
    zero
        Will stop the optimization when this cost function value is reached.
    """

    def __init__(self,
                 _opt_id: int = None,
                 _signal_pipe: Connection = None,
                 _results_queue: Queue = None,
                 _pause_flag: Event = None,
                 workers: int = 1,
                 backend: str = 'processes',
                 is_log_detailed: bool = False,
                 optimizer: str = 'TBPSA',
                 zero: float = -float('inf')):
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, workers, backend, is_log_detailed)

        self.opt_algo = ng.optimizers.registry[optimizer]
        self.optimizer = None
        if self.opt_algo.no_parallelization is True:
            warnings.warn("The selected algorithm does not support parallel execution, workers overwritten and set to"
                          " one.", RuntimeWarning)
            self.workers = 1
        self.zero = zero
        self.stop = False
        self.ng_callbacks = None

    def minimize(self, function, x0, bounds, callbacks=None, **kwargs) -> MinimizeResult:
        lower, upper = np.transpose(bounds)
        parametrization = ng.p.Array(init=x0)
        parametrization.set_bounds(lower, upper)

        if self.is_restart and self.optimizer:
            self.ng_callbacks.parent = self
            self.ng_callbacks.callbacks = callbacks
            self.stop = False
            self.logger.debug("Loaded nevergrad optimizer")
        else:
            self.optimizer = self.opt_algo(parametrization=parametrization, budget=int(4e50),
                                           num_workers=self.workers, **kwargs)
            self.ng_callbacks = _NevergradCallbacksWrapper(self, callbacks)
            self.logger.debug("Created nevergrad optimizer object")

        self.logger.debug("Created callbacks object")
        self.optimizer.register_callback('tell', self.ng_callbacks)
        self.logger.debug("Callbacks registered with optimizer")
        if self.workers > 1:
            self.logger.debug("Executing within pool with %d workers", self.workers)
            if self._backend == 'processes':
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    opt_vec = self.optimizer.minimize(function, executor=executor, batch_mode=False)
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    opt_vec = self.optimizer.minimize(function, executor=executor, batch_mode=False)
        else:
            self.logger.debug("Executing serially.")
            opt_vec = self.optimizer.minimize(function, batch_mode=False)

        self.logger.debug("Optimization complete. Formatting into MinimizeResult instance")
        results = MinimizeResult()
        results.x = opt_vec.value
        results.fx = function(results.x)
        if results.fx < float('inf'):
            results.success = True

        return results

    def callstop(self, *args):
        self.stop = True

    def checkpoint_save(self, path: Union[Path, str], force: Optional[Set[str]] = None):
        # Remove attributes which should not be saved
        if self.ng_callbacks:
            self.ng_callbacks.parent = None
            callbacks = self.ng_callbacks.callbacks
            self.ng_callbacks.callbacks = None

        super().checkpoint_save(path, {'opt_algo', 'ng_callbacks'})

        # Restore attributes
        if self.ng_callbacks:
            self.ng_callbacks.parent = self
            self.ng_callbacks.callbacks = callbacks


class _NevergradCallbacksWrapper:
    """ Wraps all the components needed by GloMPO to be called after each iteration into a single object which can be
        registered as a nevergrad callback.
    """

    def __init__(self, parent: Nevergrad,
                 callbacks: Union[None,
                                  Callable[[ng.optimizers.base.Optimizer, Sequence[float], float], bool],
                                  Sequence[Callable[[ng.optimizers.base.Optimizer, Sequence[float], float],
                                                    bool]]] = None):
        self.parent = parent
        self.i_fcalls = 0

        if callable(callbacks):
            self.callbacks = [callbacks]
        elif callbacks is None:
            self.callbacks = []

    def __call__(self, opt: ng.optimizers.base.Optimizer, x: ng.p.Array, fx: float):

        if not self.parent.stop:
            stop_cond = None

            # Normal termination condition
            if fx >= 1e30 or fx <= self.parent.zero:
                stop_cond = f"Nevergrad termination conditions:\n" \
                            f"(fx >= 1e30) = {fx >= 1e30}\n" \
                            f"(fx <= {self.parent.zero}) = {fx <= self.parent.zero}"
            self.parent.logger.debug("Stop = %s at convergence condition", bool(stop_cond))

            # User sent callbacks
            if not stop_cond and any([cb(opt, x, fx) for cb in self.callbacks]):
                stop_cond = "Direct user callbacks"
            self.parent.logger.debug("Stop = %s at user callbacks (iter: %d)", bool(stop_cond), opt.num_tell)

            # GloMPO specific callbacks
            if self.parent._results_queue:
                self.parent._pause_signal.wait()
                self.parent.check_messages()
                if not stop_cond and self.parent.stop:
                    stop_cond = "GloMPO termination signal."
                self.parent.logger.debug("Stop = %s after message check from manager", bool(stop_cond))
                self.i_fcalls = opt.num_tell + 1
                if stop_cond:
                    self.parent.logger.debug("Stop is True so shutting down optimizer.")
                    self.parent.stop = True
                    opt._num_ask = opt.budget - 1
                    self.parent.message_manager(0, stop_cond)

    @property
    def n_iter(self):
        return self.i_fcalls
