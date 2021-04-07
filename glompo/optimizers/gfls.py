from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from multiprocessing import Event
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Queue
from typing import Callable, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
from optsam import GFLS, Hook, Logger, Reporter

from .baseoptimizer import BaseOptimizer, MinimizeResult
from ..common.namedtuples import IterationResult


class GFLSOptimizer(BaseOptimizer):

    @classmethod
    def checkpoint_load(cls: Type['BaseOptimizer'], path: Union[Path, str], opt_id: Optional[int] = None,
                        signal_pipe: Optional[Connection] = None,
                        results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                        backend: str = 'threads') -> 'BaseOptimizer':
        """ Recreates a previous instance of the optimizer suitable to continue a optimization from its previous
            state. Below is a basic implementation which should suit most optimizers, may need to be overwritten.

            Parameters
            ----------
            path: Union[Path, str]
                Path to checkpoint file from which to build from. It must be a file produced by the corresponding
                BaseOptimizer().checkpoint_save method.
            opt_id, signal_pipe, results_queue, pause_flag, workers, backend
                These parameters are the same as the corresponding ones in BaseOptimizer.__init__. These will be
                regenerated and supplied by the manager during reconstruction.
        """

    @property
    def is_restart(self):
        return self._is_restart

    def __init__(self, opt_id: Optional[int] = None, signal_pipe: Optional[Connection] = None,
                 results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                 backend: str = 'threads', logger: bool = False, verbose: bool = False,
                 other_hooks: Optional[Sequence[Hook]] = None, **gfls_algo_kwargs):
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
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)
        self.gfls = None
        self.result = None
        self.stopcond = None

        self.algo_kwargs = gfls_algo_kwargs if gfls_algo_kwargs else {}
        if 'tr_max' not in self.algo_kwargs:
            self.algo_kwargs['tr_max'] = 0.5

        self.hooks = list(other_hooks)
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
                of multiple optimizer types, GFLSOptimizer will call function.resids(x) when evaluating the function.
                The API is:
                    function.resids(x: Sequence[float]) -> Tuple[fx: float, residuals: Sequence[float]]
            x0: Sequence[float]
                Initial guess from where to begin searching.
            bounds: Sequence[Tuple[float, float]]
                Sequence of min, max tuples of the same length as x0.
            callbacks: Callable = None, **kwargs
                Callbacks are called once per iteration. They receive no arguments. If they return anything other than
                None the minimization will terminate early.
        """

        # TODO: Resume from checkpoint
        # TODO: Save to checkpoint

        def function_pack(ask, func):
            _x, _is_constrained = ask()
            _fx, _resids = func(x)
            return _x, _fx, _is_constrained, _resids

        self._wrapped_func = partial(function_pack, func=function)

        if not self.is_restart:
            self.logger.info("Setting up fresh GFLS")
            self.algo_kwargs['bounds'] = bounds
            self.gfls = GFLS(np.array(x0, dtype=float), **self.algo_kwargs)
            self.result = MinimizeResult()
            for hook in self.hooks:
                hook.before_start(self.gfls)

        # Open executor pool
        if self.workers > 1:
            self._pool_evaluator = ProcessPoolExecutor if self._backend == 'processes' else ThreadPoolExecutor
            self._pool_evaluator = self._pool_evaluator(max_workers=self.workers)

        self.logger.debug("Entering optimization loop")

        while not self.stopcond:
            x, is_constrained, fx, resids = self.get_evaluation()
            self.gfls.tell(x, is_constrained, fx)

            for hook in self.hooks:
                new_stopcond = hook.after_tell(self.gfls, self.stopcond)
                if new_stopcond:
                    self.stopcond = new_stopcond

            for callback in callbacks:
                new_stopcond = callback()
                if new_stopcond:
                    self.stopcond = new_stopcond

            if self._results_queue:
                result = IterationResult(self._opt_id, self.gfls.itell, 1, x, fx, bool(self.stopcond))
                self.push_iter_result(result)
                self.check_messages()
                self._pause_signal.wait()

        self.logger.debug("Exited optimization loop")

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
                if self.injection_frequency and self.gfls.itell - self.injection_counter > self.injection_frequency:
                    x_pack = lambda: self._incumbent, True
                    self.injection_counter = self.gfls.itell
                else:
                    x_pack = self.gfls.ask()

                future = self._pool_evaluator.submit(self._wrapped_func, x_pack)
                self._futures.add(future)
                self.gfls.state["ncall"] += 1

            done, not_done = wait(self._futures, return_when=FIRST_COMPLETED)

            result = done.pop().result()
            self._futures = done + not_done  # To prevent conditions where an evaluation may be accidentally dropped

            return result

        else:
            if self.injection_frequency and self.gfls.itell - self.injection_counter > self.injection_frequency:
                x_pack = lambda: self._incumbent, True
                self.injection_counter = self.gfls.itell
            else:
                x_pack = self.gfls.ask()

            return self._wrapped_func(x_pack)

    def callstop(self, reason: str):
        """ Signal to terminate the minimize loop while still returning a result. """

    def checkpoint_save(self, path: Union[Path, str], force: Optional[Set[str]] = None):
        """ Save current state, suitable for restarting. Path is the location for the file or folder to be constructed.
            Note that only the absolutely critical aspects of the state of the optimizer need to be saved. The manager
            will resupply multiprocessing parameters when the optimizer is reconstructed. Below is a basic
            implementation which should suit most optimizers, may need to be overwritten.

            Parameters
            ----------
            path: Union[Path, str]
                Path to file into which the object will be dumped.
            force: Optional[str]
                Set of variable names which will be forced into the dumped file. Convenient shortcut for overwriting if
                fails for a particular optimizer because a certain variable is filtered out of the data dump.
        """

    def inject(self, x: Sequence[float], fx: float):
        """ If configured to do so, the manager will share the best solution seen by any optimizer with the others
            through this method. The default is to save the iteration into the _incumbent property which the minimize
            algorithm may be able to use in some way.
        """
