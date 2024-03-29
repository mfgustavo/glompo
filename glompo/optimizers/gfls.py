from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from multiprocessing import Event
from multiprocessing.connection import Connection
from queue import Queue
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from optsam import GFLS, Hook, Logger, Reporter

from .baseoptimizer import BaseOptimizer, MinimizeResult


def _function_pack(ask, func):
    _x, _is_constrained = ask()
    _fx, _resids = func.detailed_call(_x)[:2]
    return _x, _is_constrained, _fx, _resids


class GFLSOptimizer(BaseOptimizer):
    """ Wrapper around the :class:`!optsam.GFLS` algorithm [d]_.
    Note that this class is also *stand-alone*, this means it can be used independently of the GloMPO framework.

    Parameters
    ----------
    Inherited, _opt_id _signal_pipe _results_queue _pause_flag workers backend is_log_detailed
        See :class:`.BaseOptimizer`.
    logger
        If :obj:`True` a :class:`!optsam.Logger` will be run along with the optimization and saved after the
        minimization.
    verbose
        If :obj:`True` an :class:`!optsam.Reporter` will be run along with the optimisation to print progress in
        realtime.
    other_hooks
        Any extra :class:`!optsam.Hook` instances which should be included.
    **gfls_algo_kwargs
        Keyword arguments for the optsam GFLS class. If :obj:`None`, the default arguments are used:

        ====================  ===============
        Setting               Default
        ====================  ===============
        :code:`tr_max`        :code:`0.5`
        :code:`xtol`          :code:`1e-3`
        :code:`ftol`          :code:`1e-7`
        :code:`constraints`   :code:`()`
        :code:`tr_min`        :code:`None`
        :code:`tr_scale`      :code:`0.9`
        :code:`noise_scale`   :code:`0.1`
        :code:`pop_size`      :code:`None`
        :code:`diis_mode`     :code:`"qrsvd"`
        :code:`seed`          :code:`None`
        ====================  ===============

    Notes
    -----
    :class:`GFLSOptimizer` requires residuals (differences between a training set and evaluated values) to work. Thus,
    it cannot be used on all global optimization cases. To ensure compatibility and allow simultaneous use of multiple
    optimizer types, :class:`GFLSOptimizer` will automatically use :meth:`~.BaseFunction.detailed_call` when evaluating
    the function. It is assumed that the first element of the return is the total error and the second element is the
    list of residuals. Other returns are ignored.
    """

    def __init__(self,
                 _opt_id: Optional[int] = None,
                 _signal_pipe: Optional[Connection] = None,
                 _results_queue: Optional[Queue] = None,
                 _pause_flag: Optional[Event] = None,
                 workers: int = 1,
                 backend: str = 'threads',
                 is_log_detailed: bool = False,
                 logger: bool = False,
                 verbose: bool = False,
                 other_hooks: Optional[Sequence[Hook]] = None,
                 **gfls_algo_kwargs):
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, workers, backend, is_log_detailed)
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
        self._wrapped_func = partial(_function_pack, func=function)

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
            x, is_constrained, _, resids = self.get_evaluation()
            self.gfls.tell(np.array(x), is_constrained, np.array(resids))

            for hook in self.hooks:
                new_stopcond = hook.after_tell(self.gfls, self.stopcond)
                if new_stopcond:
                    self.stopcond = new_stopcond

            if callbacks:
                new_stopcond = callbacks()
                if new_stopcond:
                    self.stopcond = new_stopcond

            if self._results_queue:
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
        """ Returns a parameter vector and its evaluation.
        Depending on the configuration of the optimizer this can be a simple serial evaluation or retrieving from a list
        of completed evaluations from a pool of asynchronous parallel evaluations.
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

        self.gfls.state["ncall"] += 1
        return self._wrapped_func(self.gfls.ask())

    def callstop(self, reason: str = "Manager termination signal"):
        self.logger.debug("Calling stop. Reason = %s", reason)
        self.stopcond = reason
        self.result.success = all([reason != cond for cond in ("GloMPO Crash", "Manager termination signal")])
