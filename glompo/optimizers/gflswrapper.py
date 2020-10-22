import collections
import os
import warnings
from multiprocessing import Event, Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import numpy as np
from optsam.algo_base import AlgoBase
from optsam.codec import BoxTanh, VectorCodec
from optsam.driver import driver
from optsam.fwrap import ResidualsWrapper
from optsam.logger import Logger
from optsam.opt_gfls import GFLS

from .baseoptimizer import BaseOptimizer, MinimizeResult
from ..common.namedtuples import IterationResult

__all__ = ("GFLSOptimizer",)


class GFLSOptimizer(BaseOptimizer):

    def __init__(self,
                 opt_id: int = None,
                 signal_pipe: Connection = None,
                 results_queue: Queue = None,
                 pause_flag: Event = None,
                 workers: int = 1,
                 backend: str = 'processes',
                 tmax: Optional[int] = None,
                 imax: Optional[int] = None,
                 fmax: Optional[int] = None,
                 verbose: int = 30,
                 save_logger: Optional[str] = None,
                 gfls_kwargs: Optional[dict] = None):
        """ Instance of the GFLS optimizer that can be used through GloMPO.

        Parameters
        ----------
        opt_id : int
            Unique ID of the optimizer.
        tmax
            Stopping condition for the wall time in seconds. The optimization will
            stop when the given time has passed after one of the iterations. The
            actual time spent may be a bit longer because an ongoing iteration
            will not be interrupted.
        imax
            Stopping condition for the number of iterations.
        fmax
            Stopping condition for the number of function calls to the wrapper.
            This condition is checked after an iteration has completed. Function
            calls needed for the initialization are not counted.
        verbose
            When zero, no screen output is printed. If non-zero, the integer
            determines the frequency of printing the header of the logger.
        save_logger
            An optional string which if provided saves the output of the logger to the filename given.
        gfls_kwargs
            Arguments passed to the setup of the GFLS class. See opt_gfls.py or documentation.
        """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)
        self.tmax = tmax
        self.imax = imax
        self.fmax = fmax
        self.verbose = verbose
        self.save_logger = save_logger
        self.vector_codec = None

        gfls_kwargs = gfls_kwargs if gfls_kwargs else {}
        gfls_kwargs['tr_max'] = 1 if 'tr_max' not in gfls_kwargs else gfls_kwargs['tr_max']
        self.algorithm = GFLS(**gfls_kwargs)

    # noinspection PyMethodOverriding
    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Union[Sequence[float], Type[Logger]],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Union[Sequence[Callable[[Logger, AlgoBase, Union[str, None]], Any]],
                                  Callable[[Logger, AlgoBase, Union[str, None]], Any]] = None) -> MinimizeResult:
        """
        Executes the task of minimizing a function with GFLS.

        Parameters
        ----------
        function : Callable[[Sequence[float]], Sequence[float]]
            Function to be minimized.

            NB: GFLS is a unique class of optimizer that requires the function being
            minimized to return a sequence of residual errors between the function values evaluated at a trial set of
            parameters and some reference values. This means it is not generally applicable to all problems.

            function must include an implementation of function.resids() which returns these residuals.

        x0 : Union[Sequence[float], Type[OptimizerLogger]]
            Initial set of starting parameters or an instance of optsam.OptimizerLogger with a saved history of at least
            one iteration.
        bounds : Sequence[Tuple[float, float]]
            Sequence of tuples of the form (min, max) which bound the parameters.
        callbacks : Union[Sequence[Callable[[Type[OptimizerLogger, AlgoBase, Union[str, None]]], Any]],
                          Callable[[OptimizerLogger, AlgoBase, Union[str, None]], Any]]
            A list of functions called after every iteration.

            If GFLS is being used through the GloMPO manager calls to send iteration results to the manager and
            check incoming signals from it are automatically added to this list. Only send functionality you want over
            and above this.

            Each callback takes three arguments: ``logger``, ``algorithm`` and ``stopcond``. The ``logger`` is the same
            as the return value of optsam.driver, except that it only contains information of iterations so far.
            The ``algorithm`` is the one given to the driver. ``stopcond`` is the stopping condition after the current
            iteration and is ``None`` when the driver should carry on. The callback returns an updated value for
            ``stopcond``. If the callback has no return value, i.e. equivalent to returning ``None``.
        """

        # noinspection PyUnresolvedReferences
        if not hasattr(function, 'resids') and \
                isinstance(function.resids, collections.Callable):
            raise NotImplementedError("GFLS requires function to include a resids() method.")

        self.logger.debug(f"GFLS minimizing with:\n"
                          f"x0 = {x0}\n"
                          f"bounds = {bounds}\n"
                          f"callbacks = {callbacks}")

        gfls_bounds = []
        for bnd in bounds:
            if bnd[0] == bnd[1]:
                raise ValueError("Min and Max bounds cannot be equal. Rather fix the value and set the variable"
                                 "inactive through the interface.")
            gfls_bounds.append(BoxTanh(bnd[0], bnd[1]))
        self.vector_codec = VectorCodec(gfls_bounds)

        if not isinstance(x0, Logger):
            for i, x in enumerate(x0):
                if x < bounds[i][0] or x > bounds[i][1]:
                    raise ValueError("x0 values outside of bounds.")

        if callable(callbacks):
            callbacks = [callbacks]
        elif callbacks is None:
            callbacks = []
        if self._results_queue:
            callbacks = callbacks + [self.check_pause_flag, self.check_messages, self.push_iter_result]

        # noinspection PyUnresolvedReferences
        fw = ResidualsWrapper(function.resids, self.vector_codec.decode)
        self.logger.debug("Starting GFLS driver.")
        logger = driver(
            fw,
            self.vector_codec.encode(x0),
            self.algorithm,
            self.tmax,
            self.imax,
            self.fmax,
            self.verbose,
            callbacks
        )
        if self.save_logger:
            if os.sep in self.save_logger:
                path, name = tuple(self.save_logger.rsplit(os.sep, 1))
                os.makedirs(path)
            else:
                name = self.save_logger
            logger.save(name)

        cond = logger.aux["stopcond"]
        success = any(cond == k for k in ["xtol", "tr_min"])
        fx = logger.get("func_best", -1)
        history = logger.get_tracks("func")[0]
        index = np.where(history == fx)[0][0]
        x = logger.get("pars", index)

        if self._signal_pipe:
            self.message_manager(0, cond)
        result = MinimizeResult()
        result.success = success
        result.x = self.vector_codec.decode(x)
        result.fx = fx

        return result

    def push_iter_result(self, logger: Logger, algorithm, stopcond: str, *args):
        i = logger.current
        x = self.vector_codec.decode(logger.get("pars"))
        fx = logger.get("func")
        fin = False if stopcond is None else True
        self.logger.debug(f"Pushing iteration: {IterationResult(self._opt_id, i, 1, x, fx, fin)}")
        self._results_queue.put(IterationResult(self._opt_id, i, 1, x, fx, fin))

    def check_messages(self, logger: Logger, algorithm, stopcond):
        conds = []
        while self._signal_pipe.poll():
            message = self._signal_pipe.recv()
            if isinstance(message, int):
                conds.append(self._FROM_MANAGER_SIGNAL_DICT[message](logger, algorithm, stopcond))
            elif isinstance(message, tuple):
                conds.append(self._FROM_MANAGER_SIGNAL_DICT[message[0]](logger, algorithm, stopcond, *message[1:]))
            else:
                warnings.warn("Cannot parse message, ignoring", RuntimeWarning)
        if any([cond is not None for cond in conds]):
            mess = ""
            for cond in conds:
                mess += f"{cond} AND "
            mess = mess[:-5]
            return mess

    def callstop(self, logger: Logger, *args):
        return "Manager Termination"

    def check_pause_flag(self, *args, **kwargs):
        self._pause_signal.wait()
