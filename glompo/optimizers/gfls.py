from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from multiprocessing import Event
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Queue
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from optsam import GFLS, Hook, Logger, Reporter

from .baseoptimizer import BaseOptimizer, MinimizeResult
from ..common.namedtuples import IterationResult


class GFLSOptimizer(BaseOptimizer):

    @property
    def is_restart(self):
        return self._is_restart

    def __init__(self, opt_id: Optional[int] = None, signal_pipe: Optional[Connection] = None,
                 results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                 backend: str = 'threads', log_path: Union[None, str, Path] = None,
                 log_opt_extras: Optional[Sequence[str]] = None, is_log_detailed: bool = False,
                 logger: bool = False, verbose: bool = False, other_hooks: Optional[Sequence[Hook]] = None,
                 **gfls_algo_kwargs):
        """
        Initialisation of the GFLS optimizer wrapper for interface with GloMPO.

        Parameters
        ----------
        logger: bool = False
            If True an optsam Logger Hook will be run along with the optimisation and saved when the class is ended.
        verbose: bool = False
            If True an optsam Reporter Hook will be run along with the optimisation to print progress in realtime.
        other_hooks: Optional[Sequence[Hook]] = None
            Any extra optsam Hook instances which should be manually configured.
        gfls_algo_kwargs
            Keyword arguments for the optsam GFLS class. If None, the default arguments are used:
            Valid settings and defaults:
                tr_max      : 0.5
                xtol        : 1e-3
                ftol        : 1e-7
                constraints : ()
                tr_min      : None
                tr_scale    : 0.9
                noise_scale : 0.1
                pop_size    : None
                diis_mode   : "qrsvd"
                seed        : None
        """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag,
                         workers, backend, log_path, log_opt_extras, is_log_detailed)
        self.gfls = None
        self.result = None
        self.stopcond = None

        self.algo_kwargs = gfls_algo_kwargs if gfls_algo_kwargs else {}
        if 'tr_max' not in self.algo_kwargs:
            self.algo_kwargs['tr_max'] = 0.5

        self.hooks = list(other_hooks) if other_hooks else []
        if logger:
            self.hooks.append(Logger())
        if verbose:
            self.hooks.append(Reporter())

        # Used to manage async evaluation
        self._pool_evaluator = None
        self._futures = set()
        self._wrapped_func = None

    def minimize(self,
                 function,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        """ Minimizes a function, given an initial list of variable values `x0`, and a list of `bounds` on the
            variable values. The `callbacks` argument allows for specific callbacks such as early stopping.

            Parameters
            ----------
            function
                GFLSOptimizer requires residuals (differences between a training set and evaluated values) to work. Thus
                it cannot be used on all global optimization cases. To ensure compatibility and allow simultaneous use
                of multiple optimizer types, GFLSOptimizer will automatically use function.detailed_call(x) when
                evaluating the function. It is assumed that the first element of the return is the total error and the
                second element is the list of residuals. Other returns are ignored.
                The API is:
                    function.detailed_call(x: Sequence[float]) -> Tuple[fx: float, residuals: Sequence[float], ...]
            x0: Sequence[float]
                Initial guess from where to begin searching.
            bounds: Sequence[Tuple[float, float]]
                Sequence of min, max tuples of the same length as x0.
            callbacks: Callable = None
                Callbacks are called once per iteration. They receive no arguments. If they return anything other than
                None the minimization will terminate early.
        """

        def function_pack(ask, func):
            _x, _is_constrained = ask()
            _fx, _resids = func.detailed_call(_x)[:2]
            return _x, _is_constrained, _fx, _resids

        self._wrapped_func = partial(function_pack, func=function)

        if not self.is_restart:
            self.logger.info("Setting up fresh GFLS")
            self.algo_kwargs['bounds'] = bounds
            self.gfls = GFLS(np.array(x0, dtype=float), **self.algo_kwargs)
            self.result = MinimizeResult()
            for hook in self.hooks:
                hook.before_start(self.gfls)
        else:
            self._pool_evaluator = None
            self._futures = set()

        # Open executor pool
        if self.workers > 1:
            self._pool_evaluator = ProcessPoolExecutor if self._backend == 'processes' else ThreadPoolExecutor
            self._pool_evaluator = self._pool_evaluator(max_workers=self.workers)

        self.logger.debug("Entering optimization loop")

        while not self.stopcond:
            x, is_constrained, fx, resids = self.get_evaluation()
            self.gfls.tell(x, is_constrained, resids)

            for hook in self.hooks:
                new_stopcond = hook.after_tell(self.gfls, self.stopcond)
                if new_stopcond:
                    self.stopcond = new_stopcond

            if callbacks:
                new_stopcond = callbacks()
                if new_stopcond:
                    self.stopcond = new_stopcond

            if self._results_queue:
                result = IterationResult(self._opt_id, self.gfls.itell, 1, x, fx, bool(self.stopcond))
                self.push_iter_result(result)
                self.check_messages()
                self._pause_signal.wait()

        self.logger.debug("Exited optimization loop")

        for hook in self.hooks:
            hook.after_stop(self.gfls, self.stopcond)

        if self._pool_evaluator:
            self._pool_evaluator.shutdown()

        if self._results_queue:
            self.logger.debug("Messaging termination to manager.")
            self.message_manager(0, f"Optimizer convergence {self.stopcond}")

        return self.result

    def get_evaluation(self) -> Tuple[Sequence[float], bool, float, Sequence[float]]:
        """ When called returns a parameter vector and its evaluation. Depending on the configuration of the optimizer
            this can be a simple serial evaluation or retrieving from a list of completed evaluations from a pool of
            asynchronous parallel evaluations.
        """

        if self.workers > 1:
            while len(self._futures) < self.workers:
                future = self._pool_evaluator.submit(self._wrapped_func, self.gfls.ask())
                self._futures.add(future)
                self.gfls.state["ncall"] += 1

            done, not_done = wait(self._futures, return_when=FIRST_COMPLETED)

            result = done.pop().result()
            self._futures = done | not_done  # To prevent conditions where an evaluation may be accidentally dropped

            return result

        else:
            return self._wrapped_func(self.gfls.ask())

    def callstop(self, reason: str = "Manager termination signal"):
        self.logger.debug(f"Calling stop. Reason = {reason}")
        self.stopcond = reason
        self.result.success = all([reason != cond for cond in ("GloMPO Crash", "Manager termination signal")])
