import warnings
from multiprocessing import Event
from multiprocessing.connection import Connection
from queue import Queue

from scipy.optimize import basinhopping, differential_evolution, dual_annealing, minimize, shgo
from typing import Callable, Sequence, Tuple

from .baseoptimizer import BaseOptimizer, MinimizeResult

__all__ = ("ScipyOptimizeWrapper",)

AVAILABLE_OPTS = {'basinhopping': basinhopping,
                  'dual_annealing': dual_annealing,
                  'shgo': shgo,
                  'differential_evolution': differential_evolution}


class ManagerShutdownSignal(Exception):
    """ Special exception to exit out of SciPy optimizers when signalled by the manager. """


class GloMPOControl:
    def __init__(self, parent: 'ScipyOptimizeWrapper'):
        self.parent = parent

    def __call__(self, *args, **kwargs):
        # GloMPO specific callbacks
        if self.parent._results_queue:
            self.parent._pause_signal.wait()
            self.parent.check_messages()
            if self.parent.stop:  # callstop called through check_messages
                stop_cond = "GloMPO termination signal."
                self.parent.logger.debug("Stop = %s after message check from manager", bool(stop_cond))
                self.parent.message_manager(0, stop_cond)
                raise ManagerShutdownSignal


class ScipyOptimizeWrapper(BaseOptimizer):
    """ Wrapper around :func:`scipy.optimize.minimize`, :func:`scipy.optimize.basinhopping`,
    :func:`scipy.optimize.differential_evolution`, :func:`scipy.optimize.shgo`, and
    :func:`scipy.optimize.dual_annealing`.

    .. warning::

       This is quite a rough wrapper around SciPy's optimizers since the code is quite impenetrable to outside code, and
       callbacks do not function consistently. Therefore, most GloMPO functionality like checkpointing and information
       sharing is not available. Users are advised to try :class:`.Nevergrad` instead which offers an interface to the
       SciPy optimizers with full GloMPO functionality.

       This optimizer is also prone to hanging in certain edge cases, thus you are advised to set `end_timeout` in the
       :class:`.GloMPOManager` to a reasonable value.

    Parameters
    ----------
    Inherited, _opt_id _signal_pipe _results_queue _pause_flag _is_log_detailed workers backend
        See :class:`.BaseOptimizer`.
    method
        Accepts :code:`'basinhopping'`, :code:`'dual_annealing'`, :code:`'differential_evolution'`, and :code:`'shgo'`
        which will run the :mod:`scipy.optimize` function of the same name. Also accepts all the allowed methods to
        :func:`scipy.optimize.minimize`.
    """

    def __init__(self,
                 _opt_id: int = None,
                 _signal_pipe: Connection = None,
                 _results_queue: Queue = None,
                 _pause_flag: Event = None,
                 _is_log_detailed: bool = False,
                 workers: int = 1,
                 backend: str = 'processes',
                 method: str = 'Nelder-Mead'):
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, _is_log_detailed, workers, backend)

        self.stop = False
        self.opt_name = method
        self.opt_meth = AVAILABLE_OPTS.get(self.opt_name, minimize)

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        warnings.filterwarnings('ignore', "Method .+ cannot handle constraints nor bounds.")
        general_opt = False

        callbacks = callbacks if callbacks else []
        try:
            callbacks = [GloMPOControl(self), *callbacks]
        except TypeError:  # Catch iteration error if only a single callback was sent
            callbacks = [GloMPOControl(self), callbacks]

        def callback(*args, **kwargs):
            ret = None
            if callbacks:
                for cb in callbacks:
                    r = cb(*args, **kwargs)
                    if r is not None:
                        ret = r
            return ret

        if self.opt_meth is not basinhopping:
            kwargs['bounds'] = bounds

        if self.opt_meth is minimize:
            kwargs['method'] = self.opt_name

        result = MinimizeResult()
        try:
            sp_result = self.opt_meth(function,
                                      x0=x0,
                                      callback=callback,
                                      **kwargs)

            try:  # Different Scipy methods return different result structures
                sp_result = sp_result.lowest_optimization_result
            except (AttributeError, UnboundLocalError):
                pass

            if self._results_queue:
                self.message_manager(0, "Optimizer convergence")

            result.x = sp_result.x
            result.fx = sp_result.fun
            if general_opt:
                result.success = sp_result.success

        except ManagerShutdownSignal:
            pass

        return result

    def callstop(self, *args):
        self.stop = True
