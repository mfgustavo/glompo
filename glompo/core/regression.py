

from typing import *

import numpy as np
import emcee
import warnings
from scipy.optimize import minimize

from .logprob import LogPosterior, LogLikelihood
from ..common.namedtuples import RegressorCacheItem


__all__ = ("DataRegressor",)


class DataRegressor:
    """ Provides a class of methods to regress optimizer data against the function (y0-c)exp(-bt) + c using both
        frequentist and Bayesian techniques.

        Attributes
        ----------
        log_post: Callable
            Function which returns the log-posterior.
        log_like: Callable
            Function returning the value of the log-likelihood.
        cache: Dict[Int, RegressorCacheItem]
            Saves previous runs of the ensemble sampler.
            The dictionary keys are optimizer IDs.
            The values are a NamedTuple of:
                1) A hash key of the data used in the last run. Only if the hash of the new data matches the previous
                   run can the answers in the cache be accessed.
                2) The MLE for each parameter.
                3) Tuple of means and standard deviations for each parameter posterior distribution.
    """

    def __init__(self):
        self.log_post = LogPosterior()
        self.log_like = LogLikelihood()
        self.cache = {}

    def estimate_mle(self, t: np.ndarray, y: np.ndarray, cache_key: int = None) -> Tuple[float, float, float]:
        """ Returns the Maximum Likelihood Estimation of the exponential function through the given data.
            If multiple attempts at optimization fail a rudimentary fitting is returned based on a least-squares linear
            fit.

            Parameters
            ----------
            t: np.ndarray
                1D array of time or iteration coordinates from the optimizer logs.
            y: np.ndarray
                1D array of function evaluations.
            cache_key: int = None
                If provided the function will look for previous valid results in the cache and return them if
                found. If not found/valid then the calculation proceeds as normal and saves its result into the cache
                using the provided cache key.

            Returns
            -------
            b_est: float
                Estimate for the decay parameter.
            c_est: float
                Estimate for the asymptote parameter.
            s_est: float
                Estimate for the noise parameter.
        """
        hash_key = hash((tuple(t), tuple(y)))
        if cache_key and cache_key in self.cache and hash_key == self.cache[cache_key].hash:
            return self.cache[cache_key].mle

        quick_fit = np.polyfit(t, np.log(y - np.min(y) + 0.01), 1)

        b_est = np.clip(-quick_fit[0], 0.0001, 4.999)
        c_est = np.clip(y[-1], None, y[0]-0.001)
        s_est = np.clip(np.mean(np.abs(y - np.exp(quick_fit[1] + quick_fit[0] * t)) / y[0]), 0.0001, 0.4999)

        # Loop for retries in the case the maximisation fails
        for i in range(10):

            # Permute the starting point if the optimization failed.
            if i > 0:
                b_est, c_est, s_est = np.array([b_est, c_est, s_est]) * np.random.uniform(0.75, 1.25, 3)

            mle = minimize(fun=lambda *args: -self.log_like(*args),
                           x0=np.array([b_est, c_est, s_est]),
                           args=(t, y),
                           method='L-BFGS-B',
                           jac=lambda *args: np.array([-self.log_like.db(*args),
                                                       -self.log_like.dc(*args),
                                                       -self.log_like.ds(*args)]),
                           bounds=((0, 5), (-np.inf, y[0]), (0.0001, 0.5)))

            if mle.success:
                if cache_key:
                    self.cache[cache_key] = RegressorCacheItem(hash_key, mle.x, None)
                return mle.x

        warnings.warn(RuntimeWarning, "Multiple attempts to find the MLE failed. Returning rough estimates. These "
                                      "numbers are unreliable.")
        if cache_key:
            self.cache[cache_key] = RegressorCacheItem(hash_key, (b_est, c_est, s_est), None)
        return b_est, c_est, s_est

    def estimate_parameters(self, t: np.ndarray, y: np.ndarray, parms: Optional[str] = None, nwalkers: int = 32,
                            nsteps: int = 5000, cache_key: Any = None) -> Union[Tuple[Tuple[float, float], ...],
                                                                                Tuple[float, float]]:
        """ Runs the sampler on the log-posterior given the data. Returns an estimate and uncertainty on each
            parameter. The data is saved in the cache if a cache_key is provided.

            The fit is done using Bayesian non-linear regression. An optimization step first finds the MLE and then
            an Affine Invariant Markov Chain Monte Carlo Ensamble algorithm (Goodman & Weare, 2010) is used to sample
            the posterior. Peturbations of the MLE are used as starting points for the walkers.

            Parameters
            ----------
            t: np.ndarray
                1D array of time or iteration coordinates from the optimizer logs.
            y: np.ndarray
                1D array of function evaluations.
            parms: str = 'all'
                The parameter/s and their uncertainties to be returned. Accepts values 'decay', 'noise' and 'asymptote'.
                Returns all if the key is not provided.
            nwalkers: int = 50
                Number of ensemble samplers to be used in a single MCMC walk
            nsteps: int = 5000
                Number of iterations for the sampler.
            cache_key: Any
                If provided the regressor will first check in its cache for the answer. The cache-key is usually the
                optimizer opt_id. If the data used for the previous calculation and this one do not match then the
                calculation is rerun and the data in the cache is updated.
        """

        param_dict = {'decay': 0,
                      'asymptote': 1,
                      'noise': 2}

        hash_key = hash((tuple(t), tuple(y)))
        if cache_key and cache_key in self.cache and hash_key == self.cache[cache_key].hash:
            posterior = self.cache[cache_key].posterior
            if posterior:
                if parms in param_dict:
                    param_key = param_dict[parms]
                    return posterior[param_key]
                return posterior

        b_mle, c_mle, s_mle = self.estimate_mle(t, y, cache_key=cache_key)

        starting_pos = np.array([b_mle, c_mle, s_mle]) * np.random.uniform([0.9, 0.5, 0.8], [1.1, 1.5, 1.2],
                                                                           (nwalkers, 3))

        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        ndim=3,
                                        log_prob_fn=self.log_post,
                                        args=(t, y))

        sampler.run_mcmc(starting_pos, nsteps)

        samples = sampler.get_chain(discard=1000, flat=True)

        means = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)

        if cache_key:
            self.cache[cache_key] = RegressorCacheItem(hash_key, (b_mle, c_mle, s_mle),
                                                       tuple(tuple(pair) for pair in np.transpose([means, std])))

        if parms in param_dict:
            key = param_dict[parms]
            return means[key], std[key]
        else:
            return map(tuple, np.transpose([means, std]))
