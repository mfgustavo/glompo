

import warnings
from typing import *
from concurrent.futures import ThreadPoolExecutor

import nevergrad as ng
import numpy as np
from multiprocessing.connection import Connection
from multiprocessing import Queue, Event

from .baseoptimizer import BaseOptimizer, MinimizeResult
from ..common.namedtuples import IterationResult


__all__ = ('Nevergrad',)


class Nevergrad(BaseOptimizer):
    """ Provides access to the optimizers available through the `nevergrad` package.
        For more information see the module's `documentation <https://facebookresearch.github.io/nevergrad/>`.
    """

    def __init__(self, opt_id: int = None, signal_pipe: Connection = None, results_queue: Queue = None,
                 pause_flag: Event = None, optimizer: str = 'TBPSA', zero: float = -float('inf'),
                 **optkw):
        """
        Parameters
        ----------
        optimizer: str
            String key to the desired optimizer. See nevergrad documentation for a list of available algorithms.
        zero : float
            Will stop the optimization when this cost function value is reached.
        optkw
            Additional kwargs for the optimizer initialization.
        """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag)

        self.opt_algo = ng.optimizers.registry[optimizer]
        self.zero = zero
        self.stop = False
        self.kwargs = optkw

    def minimize(self, function, x0, bounds, callbacks=None, workers=1) -> MinimizeResult:
        if self.opt_algo.no_parallelization is True:
            workers = 1

        lower, upper = np.transpose(bounds)
        parametrization = ng.p.Array(init=x0)
        parametrization.set_bounds(lower, upper)

        optimizer = self.opt_algo(parametrization=parametrization, budget=int(4e5), num_workers=workers, **self.kwargs)
        self.logger.debug("Created nevergrad optimizer object")

        ng_callbacks = _NevergradCallbacksWrapper(self, callbacks)
        self.logger.debug("Created callbacks object")
        optimizer.register_callback('tell', ng_callbacks)
        self.logger.debug("Callbacks registered with optimizer")
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                self.logger.debug("Starting minimization in the thread pool.")
                opt_vec = optimizer.minimize(function, executor=executor, batch_mode=False)
        else:
            self.logger.debug("Starting minimization outside of the thread pool")
            opt_vec = optimizer.minimize(function, batch_mode=False)

        self.logger.debug("Optimization complete. Formatting into MinimizeResult instance")
        results = MinimizeResult()
        results.x = opt_vec.value
        results.fx = function(results.x)
        if results.fx < float('inf'):
            results.success = True

        return results

    def push_iter_result(self, iter_res: IterationResult):
        self._results_queue.put(iter_res)

    def callstop(self, *args):
        self.stop = True

    def save_state(self, *args):
        warnings.warn("Nevergrad save_state not yet implemented.", NotImplementedError)


class _NevergradCallbacksWrapper:

    """ Wraps all the components needed by GloMPO to be called after each iteration into a single object which can be
        registered as a nevergrad callback.
    """

    def __init__(self, parent: Nevergrad,
                 callbacks: Union[None,
                                  Callable[[ng.optimizers.base.Optimizer, Sequence[float], float], bool],
                                  Sequence[Callable[[ng.optimizers.base.Optimizer, Sequence[float], float],
                                                    bool]]] = None):
        """ Passes pipe and queue references into the algorithm iteration. """
        self.parent = parent
        self.i_fcalls = 0
        self.in_glompo = bool(self.parent._results_queue)

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
            self.parent.logger.debug(f"Stop = {bool(stop_cond)} at convergence condition")

            # User sent callbacks
            if not stop_cond and any([cb(opt, x, fx) for cb in self.callbacks]):
                stop_cond = "Direct user callbacks"
            self.parent.logger.debug(f"Stop = {bool(stop_cond)} at user callbacks (iter: {opt.num_tell})")

            # GloMPO specific callbacks
            if self.in_glompo:
                self.parent._pause_signal.wait()
                self.parent.check_messages()
                if not stop_cond and self.parent.stop:
                    stop_cond = f"GloMPO termination signal."
                self.parent.logger.debug(f"Stop = {bool(stop_cond)} after message check from manager")
                self.parent.logger.debug(f"Pushing result to queue")
                self.parent.push_iter_result(
                    IterationResult(opt_id=self.parent._opt_id,
                                    n_iter=opt.num_tell + 1,
                                    i_fcalls=self.i_fcalls - opt.num_tell + 1,
                                    x=x.value,
                                    fx=fx,
                                    final=bool(stop_cond)))
                self.parent.logger.debug("Result pushed successfully")
                self.i_fcalls = opt.num_tell + 1
                if stop_cond:
                    self.parent.logger.debug(f"Stop is True so shutting down optimizer.")
                    self.parent.stop = True
                    opt._num_ask = opt.budget - 1
                    self.parent.message_manager(0, stop_cond)
