import warnings
from multiprocessing import Event
from multiprocessing.connection import Connection
from queue import Queue
from typing import Callable, Sequence, Tuple

from scipy.optimize import basinhopping, differential_evolution, dual_annealing, minimize, shgo

from .baseoptimizer import BaseOptimizer, MinimizeResult

__all__ = ("ScipyOptimizeWrapper",)

AVAILABLE_OPTS = {'basinhopping': basinhopping,
                  'dual_annealing': dual_annealing,
                  'shgo': shgo,
                  'differential_evolution': differential_evolution}


class ScipyOptimizeWrapper(BaseOptimizer):
    """ Wrapper around :func:`scipy.optimize.minimize`, :func:`scipy.optimize.basinhopping`,
    :func:`scipy.optimize.differential_evolution`, :func:`scipy.optimize.shgo`, and
    :func:`scipy.optimize.dual_annealing`.

    .. warning::

       This is quite a rough wrapper around SciPy's optimizers since the code is quite impenetrable to outside code, and
       callbacks do not function consistently. Therefore, most GloMPO functionality like checkpointing and information
       sharing is not available. Users are advised to try :class:`.Nevergrad` instead which offers an interface to the
       SciPy optimizers with full GloMPO functionality.

    .. attention::

       **Must** be used with :attr:`.GloMPOManager.aggressive_kill` as :obj:`True`, this also implies that this
       optimizer can **only** be used with a multiprocessing backend; threads are incompatible.

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

        self.opt_name = method
        self.opt_meth = AVAILABLE_OPTS.get(self.opt_name, minimize)

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        warnings.filterwarnings('ignore', "Method .+ cannot handle constraints nor bounds.")
        general_opt = False

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

        sp_result = self.opt_meth(function,
                                  x0=x0,
                                  callback=callback,
                                  **kwargs)
        try:
            sp_result = sp_result.lowest_optimization_result
        except AttributeError:
            pass

        if self._results_queue:
            self.message_manager(0, "Optimizer convergence")

        result = MinimizeResult()
        result.x = sp_result.x
        result.fx = sp_result.fun
        if general_opt:
            result.success = sp_result.success

        return result

    def callstop(self, *args):
        pass
