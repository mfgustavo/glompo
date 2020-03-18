import warnings
import os
import numpy as np
import cma
import pickle
from typing import *
from time import time
from multiprocessing import Event, Queue
from multiprocessing.connection import Connection

from .baseoptimizer import MinimizeResult, BaseOptimizer
from ..common.namedtuples import IterationResult


__all__ = ('CMAOptimizer',)


class CMAOptimizer(BaseOptimizer):
    """
    Implementation of the Covariance Matrix Adaptation Evolution Strategy
        * Home:   http://cma.gforge.inria.fr/
        * Module: https://github.com/CMA-ES/pycma

    Parameters:

    sigma : float
        Standard deviation for all parameters. Parameters must be scaled accordingly!
    sampler : str, either 'vkd', 'vd' or other
        Allows the use of `GaussVDSampler` and `GaussVkDSampler` settings. Usually faster, but less effective.
    cmasettings : dict
        cma module-specific settings as ``k,v`` pairs. See ``cma.s.pprint(cma.CMAOptions())`` for a list of available
        options. Most useful keys are: `timeout`, `tolstagnation`, `popsize`. Additionally, the key `minsigma` is
        supported: Termination if ``sigma < minsigma``.
    """

    needscaler = True

    def __init__(self, opt_id: int = None, signal_pipe: Connection = None, results_queue: Queue = None,
                 pause_flag: Event = None, sigma=0.5, sampler='full', **cmasettings):
        """ Initialize with the above parameters. """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag)
        self.sigma = sigma

        # Sort all non-native CMA options into the custom cmaoptions key 'vv':
        customkeys = [i for i in cmasettings.keys() if i not in cma.CMAOptions().keys()]
        customopts = {i: cmasettings[i] for i in customkeys}
        cmasettings = {k: v for k, v in cmasettings.items() if not any(k == i for i in customkeys)}
        cmasettings['vv'] = customopts
        if 'verbose' not in cmasettings:
            cmasettings['verbose'] = -3
        self.opts = cmasettings

        if sampler == 'vd':
            self.opts = cma.restricted_gaussian_sampler.GaussVDSampler.extend_cma_options(self.opts)
        if sampler == 'vkd':
            self.opts = cma.restricted_gaussian_sampler.GaussVkDSampler.extend_cma_options(self.opts)

        self.dir = None
        self.es = None
        self.result = None

    def minimize(self, function, x0, bounds, callbacks=None,  **kwargs):
        self.dir = os.path.abspath('') + os.sep + 'cmadata' + os.sep
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)

        self.opts.update({'verb_filenameprefix': self.dir+'cma_'})

        bmin, bmax = np.transpose(bounds)
        bounds = [bmin, bmax]

        self.opts.update({'bounds': bounds})
        es = cma.CMAEvolutionStrategy(x0, self.sigma, self.opts)
        self.es = es
        self.result = MinimizeResult()

        t_start = time()
        solutions = 0
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [function(x) for x in solutions])
            es.logger.add()
            self.result.x, self.result.fx = es.result[:2]
            if self._results_queue:
                self.push_iter_result(es.countiter, len(solutions), self.result.x, self.result.fx, False)
                self.check_messages()
                self._pause_signal.wait()
            self._customtermination(callbacks)
            print(f'At CMA Iteration: {es.countiter}. Best f(x)={es.best.f:.3e}.')
            if es.countiter % 20 == 0:
                print('[DEBUG] Avg. time per cmaes loop: {}.'.format((t_start - time())/es.countiter))

        self.result.success = True
        self.result.x, self.result.fx = es.result[:2]
        if self._results_queue:
            self.push_iter_result(es.countiter, len(solutions), self.result.x, self.result.fx, True)
            self.check_messages()
        with open(self.dir+'cma_results.pkl', 'wb') as f:
            pickle.dump(es.result, f, -1)
        return self.result

    def _customtermination(self, callbacks):
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

        if callbacks and callbacks():
            self.callstop()

    def push_iter_result(self, i, f_calls, x, fx, final):
        self._results_queue.put(IterationResult(self._opt_id, i, f_calls, x, fx, final))

    def callstop(self, reason=None):
        if reason:
            print(reason)
        self.es.callbackstop = 1

    def save_state(self, *args):
        warnings.warn("CMA save_state not yet implemented.", NotImplemented)
        pass
