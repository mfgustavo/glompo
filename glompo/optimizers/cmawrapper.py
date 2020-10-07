""" Implementation of CMA-ES as a GloMPO compatible optimizer.
        Adapted from:   SCM ParAMS
        Authors:        Robert Rüger, Leonid Komissarov
"""

import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Event, Queue
from multiprocessing.connection import Connection
from typing import Callable, Optional, Sequence, Tuple

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

    def __init__(self, sigma: float = None, sampler: str = 'full', verbose: bool = True, keep_files: bool = False,
                 opt_id: Optional[int] = None, signal_pipe: Optional[Connection] = None,
                 results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                 backend: str = 'processes', restart_file: Optional[str] = None,
                 **cmasettings):
        """ Parameters
            ----------
            sigma: float
                Standard deviation for all parameters (all parameters must be scaled accordingly).
                Defines the search space as the std dev from an initial x0. Larger values will sample a wider Gaussian.
                If a restart_file is not provided this argument must be given.
            sampler: Literal['full', 'vkd', 'vd'] = 'full'
                Allows the use of `GaussVDSampler` and `GaussVkDSampler` settings.
            verbose: bool = True
                Be talkative (1) or not (0)
            workers: int = 1
                The number of parallel evaluators used by this instance of CMA.
            keep_files: bool = False
                If True the files produced by CMA are retained otherwise they are deleted. Deletion is the default
                behaviour since, when using GloMPO, GloMPO log files are created. Note, however, that GloMPO log files
                are different from CMA ones.
            cmasettings: Optional[Dict[str, Any]]
                cma module-specific settings as ``k,v`` pairs. See ``cma.s.pprint(cma.CMAOptions())`` for a list of
                available options. Most useful keys are: `timeout`, `tolstagnation`, `popsize`. Additionally,
                the key `minsigma` is supported: Termination if ``sigma < minsigma``.
        """
        assert bool(sigma) ^ bool(restart_file), "Must supply a sigma value OR a restart_file"
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend, restart_file)

        if restart_file:
            self.load_state(restart_file)
            return

        self.sigma = sigma
        self.verbose = verbose
        self.es = None
        self.result = None
        self.keep_files = keep_files
        self.folder_name = 'cmadata' if not self._opt_id else f'cmadata_{self._opt_id}'
        self.cmasettings = cmasettings

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

        if self.workers > 1 and 'popsize' not in self.cmasettings:
            self.cmasettings['popsize'] = self.workers
        self.popsize = self.cmasettings['popsize']

        if sampler == 'vd':
            self.cmasettings = GaussVDSampler.extend_cma_options(self.cmasettings)
        elif sampler == 'vkd':
            self.cmasettings = GaussVkDSampler.extend_cma_options(self.cmasettings)

        self.cmasettings.update({'verb_filenameprefix': self.folder_name + 'cma_'})

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:

        if not self._restart_file:
            self.logger.info("Setting up fresh CMA")

            os.makedirs(self.folder_name, exist_ok=True)

            self.result = MinimizeResult()
            self.cmasettings.update({'bounds': np.transpose(bounds).tolist()})
            self.es = cma.CMAEvolutionStrategy(x0, self.sigma, self.cmasettings)

        self.logger.debug("Entering optimization loop")

        i = self.es.countiter
        x = None
        while not self.es.stop():
            i += 1
            self.logger.debug("Asking for parameter vectors")
            x = self.es.ask()
            self.logger.debug("Parameter vectors generated")

            if self.workers > 1:
                if self._backend == 'processes':
                    self.logger.debug(f"Executing within process pool with {self.workers} workers")
                    with ProcessPoolExecutor(max_workers=self.workers) as executor:
                        fx = list(executor.map(function, x))
                else:
                    self.logger.debug(f"Executing within thread pool with {self.workers} workers")
                    with ThreadPoolExecutor(max_workers=self.workers) as executor:
                        fx = list(executor.map(function, x))
            else:
                self.logger.debug("Executing serially")
                fx = [function(i) for i in x]

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

            if self._results_queue:
                i_best = np.argmin(fx)
                result = IterationResult(self._opt_id, self.es.countiter, self.popsize, x[i_best], fx[i_best],
                                         self.es.stop())
                self.push_iter_result(result)
                self.logger.debug("Pushed result to queue")
                self.check_messages()
                self.logger.debug("Checked messages")
                self._pause_signal.wait()
                self.logger.debug("Passed pause test")
            self._customtermination()
            if callbacks and callbacks():
                self.callstop("Callbacks termination.")
            self.logger.debug("callbacks called")

        self.logger.debug("Exited optimization loop")

        self.result.x, self.result.fx = self.es.result[:2]
        self.result.success = np.isfinite(self.result.fx)
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
                with open(os.path.join(self.folder_name, 'cma_results.pkl'), 'wb') as file:
                    self.logger.debug("Pickling results")
                    pickle.dump(self.es.result, file)
            else:
                shutil.rmtree(self.folder_name, ignore_errors=True)

        return self.result

    def _customtermination(self):
        if 'tolstagnation' in self.cmasettings:
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

            if self._stagnationcounter > self.cmasettings['tolstagnation']:
                self.callstop("Early CMA stop: 'tolstagnation'.")

        opts = self.cmasettings['vv']
        if 'minsigma' in opts and self.es.sigma < opts['minsigma']:
            # Stop if sigma falls below minsigma
            self.callstop("Early CMA stop: 'minsigma'.")

    def callstop(self, reason="Manager termination signal"):
        if reason and self.verbose:
            print(reason)
        self.logger.debug(f"Calling stop. Reason = {reason}")
        self.es.callbackstop = 1

    def save_state(self, path: str):
        self.logger.debug("Creating restart file.")

        dump_collection = {}
        for var in dir(self):
            if not callable(getattr(self, var)) and not var.startswith('_') and var != 'logger':
                dump_collection[var] = getattr(self, var)

        with open(path, 'wb') as file:
            pickle.dump(dump_collection, file)

        self.logger.info("Restart file created successfully.")

    def load_state(self, path: str):
        self.logger.info("Initialising from restart file.")

        with open(path, 'rb') as file:
            state = pickle.load(file)

        for var, val in state.items():
            self.__setattr__(var, val)

        self.logger.info("Successfully loaded.")
