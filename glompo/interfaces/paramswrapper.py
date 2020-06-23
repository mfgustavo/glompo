

from typing import *
import warnings

import numpy as np

from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
from scm.params.core.lossfunctions import SSE

from ..core.manager import GloMPOManager
from ..opt_selectors.baseselector import BaseSelector
from ..optimizers.gflswrapper import GFLSOptimizer


__all__ = ("GlompoParamsWrapper",)


class _FunctionWrapper:
    """ Wraps function to match the API required by GloMPO 'tasks'. Can be modified to achieve compatibility with
        other optimizers.

        Currently:
        1) Returns a float from the __call__ function;
        2) Add a resids parameter for compatibility with optsam GFLS algorithm.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, pars) -> float:
        result = self.func(pars)
        fx = result[0].fx
        return fx

    def resids(self, pars):
        """ Method added to conform the function to optsam API and allow the GFLS algorithm to be used. """

        result = self.func(pars)[0]

        resids = result.residuals
        dataset = result.dataset

        if len(resids) > 0:
            weights = dataset.get('weight')
            sigmas = dataset.get('sigma')

            resids = np.concatenate([(w/s)*r for w, s, r in zip(weights, sigmas, resids)])
            print(resids)
            return resids
        else:
            return np.array([np.inf])


class GlompoParamsWrapper(BaseOptimizer):
    """ Wraps the GloMPO manager into a ParAMS optimizer. """

    def __init__(self, optimizer_selector: BaseSelector, **manager_kwargs):
        """ Accepts GloMPO configuration information.

            Parameters
            ----------
            optimizer_selector: BaseSelector
                Initialised BaseSelector object which specifies how optimizers are selected and initialised. See
                glompo.opt_selectors.BaseSelector for detailed documentation.
            **manager_kwargs
                Optional arguments to the GloMPOManager initialisation function.
                Note that all arguments are accepted but required GloMPO arguments 'task', 'n_parms',  and 'bounds'
                will be overwritten as they are passed by the 'minimize' function in accordance with ParAMS API.
        """

        self.manager_kwargs = manager_kwargs
        for kw in ['task', 'n_parms', 'bounds']:
            if kw in self.manager_kwargs:
                del self.manager_kwargs[kw]

        self.selector = optimizer_selector

        if GFLSOptimizer in optimizer_selector:
            self._loss = SSE()

    def minimize(self,
                 function: Callable,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 workers: int = 1,
                 callbacks: List[Callable] = None) -> MinimizeResult:
        """
        Passes 'function' to GloMPO to be minimized. Returns and instance of MinimizeResult.

        Parameters
        ----------
        function: Callable
            Function to be minimized, this is passed as GloMPO's 'task' parameter.
        x0: Sequence[float]
            The length of this vector is taken to be the number of parameters in the optimization. It is not, however,
            used as the starting point for any optimizer the correct way to control this is by using GloMPO
            'BaseGenerator' objects.
        bounds: Sequence[Tuple[float, float]]
            Sequence of (min, max) pairs used to bound the search area for every parameter.
            The 'bounds' parameter is passed to GloMPO as its 'bounds' parameter.

            Note that by default ParAMS shifts and scales all parameters to the interval (0, 1). GloMPO will work in
            this space and be blind to the true bounds, thus results from the GloMPO logs cannot be applied directly
            to the function.
        workers: int
            Represents the maximum number of optimizers run in parallel. Passed to GloMPO as its 'max_jobs' parameter
            if 'max_jobs' has not been sent during initialisation via manager_kwargs otherwise ignored. If allowed to
            default this will usually result in the number of optimizers as there are cores available.
        callbacks: List[Callable]
            GloMPO ignores the callbacks parameter as it is ambiguous in this context. To control the termination of
            the manager itself use BaseChecker objects passed to GloMPO's convergence_criteria parameter.

            To control individual optimizers with callbacks, pass callback functions through BaseSelector objects to
            GloMPO's optimizer_selector parameter. Callbacks in this sense, however, are discouraged as it defeats
            the purpose of GloMPO's monitored optimization.

            In some cases it may be preferable to pass callback conditions as BaseHunter objects instead.

        Notes
        -----
        GloMPO is not currently compatible with using multiple DataSets and only the first one will be considered.

        Beware of using batching with GFLS as it requires all contributions to be evaluated every iteration.

        """

        warnings.warn("The x0 parameter is ignored by GloMPO. To control the starting locations of optimizers within "
                      "GloMPO make use of its BaseGenerator objects.", RuntimeWarning)
        if callbacks:
            warnings.warn("Callbacks provided to the minimize function are ignored. Callbacks to individual "
                          "optimizers can be passed to GloMPO through BaseSelector objects. Callbacks to control the "
                          "manager itself are passed using GloMPO BaseChecker objects.")

        if 'max_jobs' not in self.manager_kwargs:
            self.manager_kwargs['max_jobs'] = workers

        manager = GloMPOManager(task=_FunctionWrapper(function),
                                n_parms=len(x0),
                                optimizer_selector=self.selector,
                                bounds=bounds,
                                **self.manager_kwargs)

        result = manager.start_manager()

        # Reshape glompo.common.namedtuples.Result into scm.params.optimizers.base.MinimizeResult
        params_res = MinimizeResult()
        params_res.x = result.x
        params_res.fx = result.fx
        params_res.success = manager.converged

        return params_res
