""" Implementation of CMA-ES as a GloMPO compatible optimizer.
        Adapted from:   SCM ParAMS
        Authors:        Robert Rüger, Leonid Komissarov
"""
import copy
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
from ..common.namedtuples import IterationResult

__all__ = ('CMAOptimizer',)


class CMAOptimizer(BaseOptimizer):
    """
    Implementation of the Covariance Matrix Adaptation Evolution Strategy
        * Home:   http://cma.gforge.inria.fr/
        * Module: https://github.com/CMA-ES/pycma
    """

    @classmethod
    def checkpoint_load(cls, path: Union[Path, str], opt_id: Optional[int] = None,
                        signal_pipe: Optional[Connection] = None,
                        results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                        backend: str = 'threads') -> 'CMAOptimizer':

        opt = cls.__new__(cls)
        super(CMAOptimizer, opt).__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)

        with open(path, 'rb') as file:
            state = pickle.load(file)

        for var, val in state.items():
            opt.__setattr__(var, val)
        opt.restart = True

        opt.logger.info("Successfully loaded from restart file.")
        return opt

    def __init__(self, sampler: str = 'full', verbose: bool = True, keep_files: bool = False,
                 opt_id: Optional[int] = None, signal_pipe: Optional[Connection] = None,
                 results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                 backend: str = 'threads', **cmasettings):
        """ Initialises the optimizer. It is built in such a way that it minimize can be called multiple times on
            different functions.

            Parameters
            ----------
            sampler: Literal['full', 'vkd', 'vd'] = 'full'
                Allows the use of `GaussVDSampler` and `GaussVkDSampler` settings.
            verbose: bool = True
                Be talkative (1) or not (0)
            workers: int = 1
                The number of parallel evaluators used by this instance of CMA.
            keep_files: bool = False
                If True the files produced by CMA are retained otherwise they are dumped in a temporary directory and
                dumped. Deletion is the default behaviour since, when using GloMPO, GloMPO log files are created.
                Note, however, that GloMPO log files are different from CMA ones.
            cmasettings: Optional[Dict[str, Any]]
                cma module-specific settings as ``k,v`` pairs. See ``cma.s.pprint(cma.CMAOptions())`` for a list of
                available options. Most useful keys are: `timeout`, `tolstagnation`, `popsize`. Additionally,
                the key `minsigma` is supported: Termination if ``sigma < minsigma``.
        """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)

        self.verbose = verbose
        self.es = None
        self.result = None
        self.keep_files = keep_files
        self.cmasettings = cmasettings
        self.restart = False
        self.popsize = cmasettings['popsize'] if 'popsize' in cmasettings else None

        # Sort all non-native CMA options into the custom cmaoptions key 'vv':
        customopts = {}
        for key, val in [*self.cmasettings.items()]:
            if key not in cma.CMAOptions().keys():
                customopts[key] = val
                del self.cmasettings[key]

        self.cmasettings['vv'] = customopts
        self.cmasettings['verbose'] = -3  # Silence CMA Logger
        if 'tolstagnation' not in self.cmasettings:
            self.cmasettings['tolstagnation'] = int(1e22)
        if 'maxiter' not in self.cmasettings:
            self.cmasettings['maxiter'] = float('inf')
        if 'tolfunhist' not in self.cmasettings:
            self.cmasettings['tolfunhist'] = 1e-15
        if 'tolfun' not in self.cmasettings:
            self.cmasettings['tolfun'] = 1e-20

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
        """ Minimize function with a starting distribution defined by mean x0 and standard deviation sigma0. Parameters
            are bounded by bounds. Callbacks are executed once per iteration.

            Parameters
            ----------
            function: Callable[[Sequence[float]], float]
                Task to be minimised accepting a sequence of unknown parameters and returning a float result.
            x0: Sequence[float]
                Initial mean of the distribution from with trial parameter sets are sampled.
            bounds: Sequence[Tuple[float, float]]
                Bounds on the sampling limits of each dimension. CMA supports handling bounds as non-linear
                transformations so that they are never exceeded (but bounds are likely to be over sampled) or with a
                penalty function. See cma documentation for more on this.
            callbacks: Callable[[], Optional[Any]]
                Callbacks are called once per iteration. They receive no arguments. If they return anything other than
                None the minimization will terminate early.
            sigma0: float = 0
                Standard deviation for all parameters (all parameters must be scaled accordingly).
                Defines the search space as the std dev from an initial x0. Larger values will sample a wider
                Gaussian. Default is zero which will not accepted by the optimizer, thus this argument must be provided.

            Notes
            -----
            If 'popsize' is not provided in the cmasettings directory at init, it will be set to the number of workers
            (if this is larger than 1) or failing that it will be set to the default 4 + int(3 * log(d)).
        """
        task_settings = copy.deepcopy(self.cmasettings)

        if not self.popsize:
            if self.workers > 1:
                task_settings['popsize'] = self.workers
            else:
                task_settings['popsize'] = 4 + int(3 * np.log(len(x0)))
        self.popsize = task_settings['popsize']

        if self.popsize < self.workers:
            warnings.warn(f"'popsize'={self.popsize} is less than 'workers'={self.workers}. "
                          f"This is an inefficient use of resources")
            self.logger.warning(f"'popsize'={self.popsize} is less than 'workers'={self.workers}. "
                                f"This is an inefficient use of resources")

        if not self.restart:
            self.logger.info("Setting up fresh CMA")

            self.result = MinimizeResult()
            task_settings.update({'bounds': np.transpose(bounds).tolist()})
            self.es = cma.CMAEvolutionStrategy(x0, sigma0, task_settings)

        self.logger.debug("Entering optimization loop")

        i = self.es.countiter
        x = None
        while not self.es.stop():
            i += 1
            self.logger.debug("Asking for parameter vectors")
            x = self.es.ask()
            self.logger.debug("Parameter vectors generated")

            if self.workers > 1:
                pool_executor = ProcessPoolExecutor if self._backend == 'processes' else ThreadPoolExecutor
                self.logger.debug(f"Executing within {pool_executor.__name__} with {self.workers} workers")
                with pool_executor(max_workers=self.workers) as executor:
                    submitted = {slot: executor.submit(function, parms) for slot, parms in enumerate(x)}
                    # For very slow evaluations this will allow evaluations to be interrupted.
                    if self._results_queue:
                        loop = 0
                        for _ in as_completed(submitted.values()):
                            loop += 1
                            self.logger.debug(f"Result {loop}/{self.popsize} returned.")
                            self._pause_signal.wait()
                            self.check_messages()
                            if self.es.callbackstop == 1:
                                self.logger.debug("Stop command received during function evaluations.")
                                cancelled = [future.cancel() for future in submitted.values()]
                                self.logger.debug(f"Aborted {sum(cancelled)} calls.")
                                break
                    fx = [future.result() for future in submitted.values() if not future.cancelled()]
            else:
                self.logger.debug("Executing serially")
                fx = [function(i) for i in x]

            if len(x) != len(fx):
                self.logger.debug("Unfinished evaluation detected. Breaking out of loop")
                break

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
                i_best = np.argmin(fx)
                result = IterationResult(self._opt_id, self.es.countiter, self.popsize, x[i_best], fx[i_best],
                                         bool(self.es.stop()))
                self.push_iter_result(result)
                self.logger.debug("Pushed result to queue")
                self.check_messages()
                self.logger.debug("Checked messages")
                self._pause_signal.wait()
                self.logger.debug("Passed pause test")
            self._customtermination(task_settings)
            self.logger.debug("callbacks called")

        self.logger.debug("Exited optimization loop")

        self.result.x, self.result.fx = self.es.result[:2]
        self.result.success = np.isfinite(self.result.fx) and self.result.success
        if self.result.fx == float('inf'):
            self.logger.warning("CMA iteration found no valid results."
                                "fx = 'inf' and x = (first vector generated by es.ask())")
            self.result.x = x[0]

        if self.verbose:
            print(f"Optimization terminated: success = {self.result.success}")
            print(f"Final fx={self.result.fx:.2E}")

        if self._results_queue:
            self.logger.debug("Messaging termination to manager.")
            self.message_manager(0, f"Optimizer convergence {self.es.stop()}")

        if self.es.stop() != "Checkpoint Shutdown":
            if self.keep_files:
                name = 'cma_'
                if self._opt_id:
                    name += f'opt{self._opt_id}_'
                name += '_results.pkl'
                with open(name, 'wb') as file:
                    self.logger.debug("Pickling results")
                    pickle.dump(self.es.result, file)

        return self.result

    def _customtermination(self, settings):
        if 'tolstagnation' in settings:
            # The default 'tolstagnation' criterium is way too complex (as most 'tol*' criteria are).
            # Here we hack it to actually do what it is supposed to do: Stop when no change after last x iterations.
            if not hasattr(self, '_stagnationcounter'):
                self._stagnationcounter = 0
                self._prev_best = self.es.best.f

            if self._prev_best == self.es.best.f:
                self._stagnationcounter += 1
            else:
                self._prev_best = self.es.best.f
                self._stagnationcounter = 0

            if self._stagnationcounter > settings['tolstagnation']:
                self.callstop("Early CMA stop: 'tolstagnation'.")

        opts = settings['vv']
        if 'minsigma' in opts and self.es.sigma < opts['minsigma']:
            # Stop if sigma falls below minsigma
            self.callstop("Early CMA stop: 'minsigma'.")

    def callstop(self, reason="Manager termination signal"):
        if reason and self.verbose:
            print(reason)
        self.logger.debug(f"Calling stop. Reason = {reason}")
        self.es.callbackstop = 1
        self.result.success = all([reason != cond for cond in ("GloMPO Crash", "Manager termination signal")])

    def checkpoint_save(self, path: Union[Path, str]):
        self.logger.debug("Creating restart file.")

        dump_collection = {}
        for var in dir(self):
            if not callable(getattr(self, var)) and not var.startswith('_') and var != 'logger':
                dump_collection[var] = getattr(self, var)
        print(f"Checkpoint at iter={self.es.countiter}")
        with open(path, 'wb') as file:
            pickle.dump(dump_collection, file)

        self.logger.info("Restart file created successfully.")
