# TODO Throw away first x number of CMA evaluations, it has terrible burn-in. How to define x?
import warnings

from .baseoptimizer import MinimizeResult, BaseOptimizer
import os
import numpy as np
import cma
import pickle
from time import time

__all__ = ['CMAOptimizer']


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

    def __init__(self, opt_id, sigma=0.5, sampler='full', **cmasettings):
        """ Initialize with the above parameters. """
        super().__init__(opt_id)
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

    def minimize(self, function, x0, bounds, callbacks=None, **kwargs):
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
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [function(x) for x in solutions])
            es.logger.add()
            if self.__results_queue:
                self.message_manager(es.countiter, self.result.x, self.result.fx)
                self.check_messages()
            self.result.x, self.result.fx = es.result[:2]
            self._customtermination(callbacks)
            print(f'At CMA Iteration: {es.countiter}. Best f(x)={es.best.f:.3e}.')
            if es.countiter % 20 == 0:
                print('[DEBUG] Avg. time per cmaes loop: {}.'.format((t_start - time())/es.countiter))
        print()

        self.result.success = True
        self.result.x, self.result.fx = es.result[:2]
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

    def message_manager(self, i, x, fx):
        self.__results_queue.put((self.__opt_id, i, x, fx))

    def callstop(self, reason=None):
        if reason:
            print(reason)
        self.es.callbackstop = 1

    def save_state(self, *args):
        warnings.warn("CMA save_state not yet implemented.", NotImplemented)
        pass
