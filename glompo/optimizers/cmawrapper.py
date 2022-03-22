""" Implementation of CMA-ES as a GloMPO compatible optimizer.
        Adapted from:   SCM ParAMS
"""
import copy
import logging
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Event, Queue
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import cma
import numpy as np
from cma.restricted_gaussian_sampler import GaussVDSampler, GaussVkDSampler

from .baseoptimizer import BaseOptimizer, MinimizeResult

__all__ = ('CMAOptimizer',)


class CMAOptimizer(BaseOptimizer):
    """ Wrapper around a CMA-ES python implementation [c]_.
    Note that this class is also *stand-alone*, this means it can be used independently of the GloMPO framework.
    It is also built in such a way that it :meth:`minimize` can be called multiple times on different functions.

    Parameters
    ----------
    Inherited, _opt_id _signal_pipe _results_queue _pause_flag _is_log_detailed workers backend
        See :class:`.BaseOptimizer`.
    sampler
        Allows the use of :code:`'GaussVDSampler'` and :code:`'GaussVkDSampler'` settings.
    verbose
        If :obj:`True`, print status messages during the optimization, else no output will be printed.
    keep_files
        Directory in which to save a file containing extra optimizer convergence information.
    force_injects
        If :obj:`True`, injections of parameter vectors into the solver will be exact, guaranteeing that that
        solution will be in the next iteration's population. If :obj:`False`, the injection will result in a direction
        relative nudge towards the vector. Forcing the injecting can limit global exploration but non-forced
        injections may have little effect.
    injection_frequency
        If :obj:`None`, injections are ignored by the optimizer. If an :obj:`int` is provided then injection are only
        accepted if at least `injection_frequency` iterations have passed since the last injection.
    **cmasettings
        `CMA-ES <http://cma.gforge.inria.fr/apidocs-pycma/>`_ package-specific settings. See
        :code:`cma.s.pprint(cma.CMAOptions())` for a list of available options. Most useful keys are: :code:`'timeout'`,
        :code:`'tolstagnation'`, :code:`'popsize'`.

    Notes
    -----
    #. Although not the default, by adjusting the injection settings above, the optimizer will inject the saved incumbent
       solution into the solver influencing the points sampled by the following iteration. The incumbent begins at
       :code:`x0` and is updated by the inject method called by the GloMPO manager.

    #. If :code:`'popsize'` is not provided during optimizer initialisation, it will be set to the number of
       :attr:`~.BaseOptimizer.workers` if this is larger than 1, else it will be set to the default:
       :code:`4 + int(3 * log(d))`.
    """

    def __init__(self,
                 _opt_id: Optional[int] = None,
                 _signal_pipe: Optional[Connection] = None,
                 _results_queue: Optional[Queue] = None,
                 _pause_flag: Optional[Event] = None,
                 _is_log_detailed: bool = False,
                 workers: int = 1,
                 backend: str = 'threads',
                 sampler: str = 'full',
                 verbose: bool = True,
                 keep_files: Union[None, str, Path] = None,
                 force_injects: Optional[bool] = None,
                 injection_frequency: Optional[int] = None,
                 **cmasettings):
        super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, _is_log_detailed, workers, backend)

        self.verbose = verbose
        self.es = None
        self.result = None
        self.keep_files = Path(keep_files) if (keep_files is not None) and (keep_files is not False) else None
        self.cmasettings = cmasettings
        self.popsize = cmasettings['popsize'] if 'popsize' in cmasettings else None

        self.force_injects = force_injects
        self.injection_frequency = injection_frequency
        self.injection_counter = 0

        # Sort all non-native CMA options into the custom cmaoptions key 'vv':
        customopts = {}
        for key, val in [*self.cmasettings.items()]:
            if key not in cma.CMAOptions().keys():
                customopts[key] = val
                del self.cmasettings[key]

        self.cmasettings['vv'] = customopts
        self.cmasettings['verbose'] = -3  # Silence CMA Logger

        # Deactivated to not interfere with GloMPO hunting
        if 'maxiter' not in self.cmasettings:
            self.cmasettings['maxiter'] = float('inf')

        if sampler == 'vd':
            self.cmasettings = GaussVDSampler.extend_cma_options(self.cmasettings)
        elif sampler == 'vkd':
            self.cmasettings = GaussVkDSampler.extend_cma_options(self.cmasettings)

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable[[], Optional[Any]] = None,
                 sigma0: float = 0, **kwargs) -> MinimizeResult:
        """ Begin CMA-ES minimization loop.

        Parameters
        ----------
        Inherited, function bounds callbacks
            See :meth:`.BaseOptimizer.minimize`
        x0
            Initial mean of the multivariate normal distribution from which trials are drawn. Force injected into the
            solver to guarantee it is evaluated.
        sigma0
            Initial standard deviation of the multivariate normal distribution from which trials are drawn. One value
            for all parameters which means that all parameters must be scaled accordingly. Default is zero which will
            not accepted by the optimizer, thus this argument must be provided.

        Returns
        -------
        MinimizeResult
            Location, function value and other optimization information about the lowest value found by the optimizer.

        Raises
        ------
        ValueError
            If `sigma0` is not changed from the default value of zero.
        """
        task_settings = copy.deepcopy(self.cmasettings)

        if sigma0 <= 0 and not self.is_restart:
            self.logger.critical('sigma0 value invalid. Please select a positive value.')
            raise ValueError('sigma0 value invalid. Please select a positive value.')

        if not self.popsize:
            if self.workers > 1:
                task_settings['popsize'] = self.workers
            else:
                task_settings['popsize'] = 4 + int(3 * np.log(len(x0)))
            self.popsize = task_settings['popsize']

        if self.popsize < self.workers:
            warnings.warn(f"'popsize'={self.popsize} is less than 'workers'={self.workers}. "
                          f"This is an inefficient use of resources")
            self.logger.warning("'popsize'=%d is less than 'workers'=%d. This is an inefficient use of resources",
                                self.popsize, self.workers)

        if not self.is_restart:
            self.logger.info("Setting up fresh CMA")

            self.result = MinimizeResult()
            task_settings.update({'bounds': np.transpose(bounds).tolist()})
            self.es = cma.CMAEvolutionStrategy(x0, sigma0, task_settings)
            self.es.inject([x0], force=True)

        self.logger.debug("Entering optimization loop")

        i = self.es.countiter
        x = None
        while not self.es.stop():
            i += 1
            self.logger.debug("Asking for parameter vectors")
            x = self.es.ask()
            self.logger.debug("Parameter vectors generated")

            fx = self._parallel_map(function, x)

            if len(x) != len(fx):
                self.logger.debug("Unfinished evaluation detected. Breaking out of loop")
                break

            if i == 1:
                self.incumbent = {'x': x0, 'fx': fx[0]}

            self.es.tell(x, fx)
            self.logger.debug("Told solutions")
            self.result.x, self.result.fx = self.es.result[:2]
            if self.result.fx == float('inf'):
                self.logger.warning("CMA iteration found no valid results."
                                    "fx = 'inf' and x = (first vector generated by es.ask())")
                self.result.x = x[0]
            self.logger.debug("Extracted x and fx from result")
            if self.verbose and i % 10 == 0 or i == 1:
                print(f"@ iter = {i} fx={self.result.fx:.2E} sigma={self.es.sigma:.3E}")

            if callbacks and callbacks():
                self.callstop("Callbacks termination.")

            if self._results_queue:
                self.check_messages()
                self.logger.debug("Checked messages")
                self._pause_signal.wait()
                self.logger.debug("Passed pause test")
            self.logger.debug("callbacks called")

            if self.incumbent['fx'] < min(fx) and \
                    self.injection_frequency and i - self.injection_counter > self.injection_frequency:
                self.injection_counter = i
                self.es.inject([self.incumbent['x']], force=self.force_injects)

        self.logger.debug("Exited optimization loop")

        self.result.x, self.result.fx = self.es.result[:2]
        self.result.success = np.isfinite(self.result.fx) and self.result.success
        if self.result.fx == float('inf'):
            self.logger.warning("CMA iteration found no valid results."
                                "fx = 'inf' and x = (first vector generated by es.ask())")
            self.result.x = x[0]

        if self.verbose:
            print(f"Optimization terminated: success = {self.result.success}")
            print(f"Optimizer convergence {self.es.stop()}")
            print(f"Final fx={self.result.fx:.2E}")

        if self._results_queue:
            self.logger.debug("Messaging termination to manager.")
            self.message_manager(0, f"Optimizer convergence {self.es.stop()}")

        if self.es.stop() != "Checkpoint Shutdown":
            if self.keep_files:
                name = 'cma_'
                if self._opt_id:
                    name += f'opt{self._opt_id}_'
                name += 'results.pkl'
                with (self.keep_files / name).open('wb') as file:
                    self.logger.debug("Pickling results")
                    pickle.dump({key: getattr(self.es, key)
                                 for key in ('result', 'sigma', 'B', 'C', 'D', 'popsize', 'sigma0')},
                                file)

        return self.result

    def _parallel_map(self, function: Callable[[Sequence[float]], float],
                      x: Sequence[Sequence[float]]) -> Sequence[float]:
        """ Returns the function evaluations for a given set of trial parameters, x.
        Calculations are distributed over threads or processes depending on the number of workers and backend selected.
        """
        if self.workers > 1:
            pool_executor = ProcessPoolExecutor if self._backend == 'processes' else ThreadPoolExecutor
            self.logger.debug("Executing within %s with %d workers", pool_executor.__name__, self.workers)
            with pool_executor(max_workers=self.workers) as executor:
                submitted = {slot: executor.submit(function, parms) for slot, parms in enumerate(x)}
                # For very slow evaluations this will allow evaluations to be interrupted.
                if self._results_queue:
                    loop = 0
                    for _ in as_completed(submitted.values()):
                        loop += 1
                        self.logger.debug("Result %d/%d returned.", loop, len(x))
                        self._pause_signal.wait()
                        self.check_messages()
                        if self.es.callbackstop == 1:
                            self.logger.debug("Stop command received during function evaluations.")
                            cancelled = [future.cancel() for future in submitted.values()]
                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug("Aborted %d calls.", sum(cancelled))
                            break
                fx = [future.result() for future in submitted.values() if not future.cancelled()]
        else:
            self.logger.debug("Executing serially")
            fx = [function(i) for i in x]
        return fx

    def callstop(self, reason: str = "Manager termination signal"):
        if reason and self.verbose:
            print(reason)
        self.logger.debug("Calling stop. Reason = %s", reason)
        self.es.callbackstop = 1
        self.result.success = all([reason != cond for cond in ("GloMPO Crash", "Manager termination signal")])
