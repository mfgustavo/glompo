""" Implementation of CMA-ES as a GloMPO compatible optimizer.
        Adapted from:   SCM ParAMS
        Authors:        Robert Rüger, Leonid Komissarov
"""

import os
import pickle
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Event, Queue
from multiprocessing.connection import Connection
from typing import Callable, Sequence, Tuple

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

    def __init__(self, opt_id: int = None, signal_pipe: Connection = None, results_queue: Queue = None,
                 pause_flag: Event = None, workers: int = 1, backend: str = 'processes',
                 sigma: float = 0.5, sampler: str = 'full', verbose: bool = True, keep_files: bool = False,
                 **cmasettings):
        """ Parameters
            ----------
            sigma: float
                Standard deviation for all parameters (all parameters must be scaled accordingly).
                Defines the search space as the std dev from an initial x0. Larger values will sample a wider Gaussian.
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
            cmasettings : Optional[Dict[str, Any]]
                cma module-specific settings as ``k,v`` pairs. See ``cma.s.pprint(cma.CMAOptions())`` for a list of
                available options. Most useful keys are: `timeout`, `tolstagnation`, `popsize`. Additionally,
                the key `minsigma` is supported: Termination if ``sigma < minsigma``.
        """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)
        self.sigma = sigma
        self.verbose = verbose
        self.sampler = sampler
        self.opts = {}
        self.dir = ''
        self.es = None
        self.result = None
        self.keep_files = keep_files

        # Sort all non-native CMA options into the custom cmaoptions key 'vv':
        customkeys = [i for i in cmasettings if i not in cma.CMAOptions().keys()]
        customopts = {i: cmasettings[i] for i in customkeys}
        cmasettings = {k: v for k, v in cmasettings.items() if not any(k == i for i in customkeys)}
        cmasettings['vv'] = customopts
        cmasettings['verbose'] = -3  # Silence CMA Logger
        if 'tolstagnation' not in cmasettings:
            cmasettings['tolstagnation'] = int(1e22)
        if 'maxiter' not in cmasettings:
            cmasettings['maxiter'] = float('inf')
        if 'tolfunhist' not in cmasettings:
            cmasettings['tolfunhist'] = 1e-15
        if 'tolfun' not in cmasettings:
            cmasettings['tolfun'] = 1e-20
        self.cmasettings = cmasettings

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:

        self.opts = self.cmasettings.copy()
        if self.workers > 1 and 'popsize' not in self.opts:
            self.opts['popsize'] = self.workers
        if self.sampler == 'vd':
            self.opts = GaussVDSampler.extend_cma_options(self.opts)
        elif self.sampler == 'vkd':
            self.opts = GaussVkDSampler.extend_cma_options(self.opts)

        folder_name = 'cmadata' if not self._opt_id else f'cmadata_{self._opt_id}'
        self.dir = os.path.abspath('.') + os.sep + folder_name + os.sep
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)

        self.opts.update({'verb_filenameprefix': self.dir + 'cma_'})
        self.opts.update({'bounds': np.transpose(bounds).tolist()})
        self.logger.debug("Updated options")

        self.result = MinimizeResult()
        self.es = es = cma.CMAEvolutionStrategy(x0, self.sigma, self.opts)

        self.logger.debug("Entering optimization loop")

        x = None
        i = 0
        while not es.stop():
            i += 1
            self.logger.debug("Asking for parameter vectors")
            x = es.ask()
            self.logger.debug("Parameter vectors generated")

            if self.workers > 1:
                self.logger.debug(f"Executing within pool with {self.workers} workers")
                if self._backend == 'processes':
                    with ProcessPoolExecutor(max_workers=self.workers) as executor:
                        fx = list(executor.map(function, x))
                else:
                    with ThreadPoolExecutor(max_workers=self.workers) as executor:
                        fx = list(executor.map(function, x))
            else:
                self.logger.debug("Executing serially")
                fx = [function(i) for i in x]

            es.tell(x, fx)
            self.logger.debug("Told solutions")
            es.logger.add()
            self.logger.debug("Add solutions to log")
            self.result.x, self.result.fx = es.result[:2]
            if self.result.fx == float('inf'):
                self.logger.warning("CMA iteration found no valid results."
                                    "fx = 'inf' and x = (first vector generated by es.ask())")
                self.result.x = x[0]
            self.logger.debug("Extracted x and fx from result")
            if self.verbose and i % 10 == 0 or i == 1:
                print(f"@ iter = {i} fx={self.result.fx:.2E}")

            if self._results_queue:
                i_best = np.argmin(fx)
                self.push_iter_result(es.countiter, len(x), x[i_best], fx[i_best], False)
                self.logger.debug("Pushed result to queue")
                self.check_messages()
                self.logger.debug("Checked messages")
                self._pause_signal.wait()
                self.logger.debug("Passed pause test")
            self._customtermination()
            if callbacks and callbacks():
                self.callstop("Callbacks termination.")
            self.logger.debug("callbacks called")

        self.logger.debug(f"Exited optimization loop")
        self.result.x, self.result.fx = es.result[:2]
        self.result.success = np.isfinite(self.result.fx)
        if self.result.fx == float('inf'):
            self.logger.warning("CMA iteration found no valid results."
                                "fx = 'inf' and x = (first vector generated by es.ask())")
            self.result.x = x[0]
        if self.verbose and i % 10 == 0 or i == 1:
            print(f"Optimization terminated: success = {self.result.success}")
            print(f"Final fx={self.result.fx:.2E}")

        if self._results_queue:
            self.logger.debug("Pushing final result")
            self.push_iter_result(es.countiter, len(x), self.result.x, self.result.fx, True)
            self.logger.debug("Messaging termination to manager.")
            self.message_manager(0, "Optimizer convergence")
            self.logger.debug("Final message check")
            self.check_messages()

        if self.keep_files:
            with open(self.dir + 'cma_results.pkl', 'wb') as f:
                self.logger.debug(f"Pickling results")
                pickle.dump(es.result, f, -1)

        if not self.keep_files:
            shutil.rmtree(folder_name, ignore_errors=True)

        return self.result

    def _customtermination(self):
        es = self.es
        if 'tolstagnation' in self.opts:
            # The default 'tolstagnation' criterium is way too complex (as most 'tol*' criteria are).
            # Here we hack it to actually do what it is supposed to do: Stop when no change after last x iterations.
            if not hasattr(self, '_stagnationcounter'):
                self._stagnationcounter = 0
                self._prev_best = es.best.f

            if self._prev_best == es.best.f:
                self._stagnationcounter += 1
            else:
                self._prev_best = es.best.f
                self._stagnationcounter = 0

            if self._stagnationcounter > self.opts['tolstagnation']:
                self.callstop("Early CMA stop: 'tolstagnation'.")

        opts = self.opts['vv']
        if 'minsigma' in opts and es.sigma < opts['minsigma']:
            # Stop if sigma falls below minsigma
            self.callstop("Early CMA stop: 'minsigma'.")

    def push_iter_result(self, i, f_calls, x, fx, final_push):
        self.logger.debug(f"Pushing result:\n"
                          f"\topt_id = {self._opt_id}\n"
                          f"\ti = {i}\n"
                          f"\tf_calls = {f_calls}\n"
                          f"\tx = {x}\n"
                          f"\tfx = {fx}\n"
                          f"\tfinal = {final_push}")
        self._results_queue.put(IterationResult(self._opt_id, i, f_calls, x, fx, final_push))

    def callstop(self, reason="Manager termination signal"):
        if reason and self.verbose:
            print(reason)
        self.logger.debug(f"Calling stop. Reason = {reason}")
        self.es.callbackstop = 1

    def save_state(self, *args):
        warnings.warn("CMA save_state not yet implemented.", NotImplementedError)
