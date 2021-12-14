import warnings
from selectors import BaseSelector
from typing import Sequence, Tuple

from scm.params.core.opt_components import _Step
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult

from ...core.manager import GloMPOManager


class _FunctionWrapper:
    """ Wraps function produced by ParAMS internals (instance of :class:`scm.params.core.opt_components._Step`) to
    match the API required by the :attr:`.GloMPOManager.task`. Can be modified to achieve compatibility with other
    optimizers.
    """

    def __init__(self, func: _Step):
        self.func = func
        if self.func.cbs:
            warnings.warn("Callbacks provided through the Optimization class are ignored. Callbacks to individual "
                          "optimizers can be passed to GloMPO through BaseSelector objects. Callbacks to control the "
                          "manager itself are passed using GloMPO BaseChecker objects, some conditions should be sent "
                          "as BaseHunter objects.", UserWarning)
            self.func.cbs = None

    def __call__(self, pars) -> float:
        return self.func(pars)


class GlompoParamsWrapper(BaseOptimizer):
    """ Wraps the GloMPO manager into a ParAMS :class:`~scm.params.optimizers.base.BaseOptimizer`.
    This is not the recommended way to make use of the GloMPO interface, it is preferable to make use of the
    :class:`.BaseParamsError` classes. This class is only applicable in cases where the ParAMS
    :class:`~scm.params.core.parameteroptimization.Optimization` class interface is preferred.

    Parameters
    ----------
    opt_selector
        Initialised :class:`.BaseSelector` object which specifies how optimizers are selected and initialised.
    **manager_kwargs
        Optional arguments to the :class:`.GloMPOManager` initialisation function.

    Notes
    -----
    `manager_kwargs` accepts all arguments of :meth:`.GloMPOManager.setup` but required GloMPO arguments
    :attr:`~.GloMPOManager.task` and :attr:`~.GloMPOManager.bounds` will be overwritten as they are passed by the
    :meth:`minimize` function in accordance with ParAMS API.
    """

    def __init__(self, opt_selector: BaseSelector, **manager_kwargs):
        self.manager = GloMPOManager()
        self.manager_kwargs = manager_kwargs
        for kw in ['task', 'bounds']:
            if kw in self.manager_kwargs:
                del self.manager_kwargs[kw]

        self.selector = opt_selector

    def minimize(self,
                 function: _Step,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 workers: int = 1) -> MinimizeResult:
        """
        Passes 'function' to GloMPO to be minimized. Returns an instance of MinimizeResult.

        Parameters
        ----------
        function
            Function to be minimized, this is passed as GloMPO's :attr:`~.GloMPOManager.task` parameter.
        x0
            Ignored by GloMPO, the correct way to control the optimizer starting points is by using GloMPO
            :class:`.BaseGenerator` objects.
        bounds
            Sequence of (min, max) pairs used to bound the search area for every parameter. The 'bounds' parameter is
            passed to GloMPO as its :attr:`~.GloMPOManager.bounds` parameter.
        workers
            Represents the maximum number of optimizers run in parallel. Passed to GloMPO as its
            :attr:`~.GloMPOManager.max_jobs` parameter if it has not been sent during initialisation via
            `manager_kwargs` otherwise ignored. If allowed to default this will usually result in the number of
            optimizers as there are cores available.

        Notes
        -----
        By default ParAMS shifts and scales all parameters to the interval (0, 1). GloMPO will work in this space and be
        blind to the true bounds, thus results from the GloMPO logs cannot be applied directly to the function.
        """

        warnings.warn("The x0 parameter is ignored by GloMPO. To control the starting locations of optimizers within "
                      "GloMPO make use of its BaseGenerator objects.", RuntimeWarning)

        if 'max_jobs' not in self.manager_kwargs:
            self.manager_kwargs['max_jobs'] = workers

        # Silence function printing
        function.v = False

        self.manager.setup(task=_FunctionWrapper(function), bounds=bounds, opt_selector=self.selector,
                           **self.manager_kwargs)

        result = self.manager.start_manager()

        # Reshape glompo.common.namedtuples.Result into scm.params.optimizers.base.MinimizeResult
        params_res = MinimizeResult()
        params_res.x = result.x
        params_res.fx = result.fx
        params_res.success = self.manager.converged and len(result.x) > 0

        return params_res

    def reset(self):
        self.manager = GloMPOManager()
