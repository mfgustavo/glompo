

# Native Python imports
from typing import *
import warnings

# AMS imports
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult

# Package imports
from ..core.manager import GloMPOManager
from ..opt_selectors.baseselector import BaseSelector


class GlompoParamsWrapper(BaseOptimizer):
    """ Wraps the GloMPO manager into a ParAMS optimizer. """

    def __init__(self, optimizer_selector: BaseSelector, manager_kwargs: Optional[Dict[str, Any]] = None):
        """ Accepts GloMPO configurational information.

            Parameters
            ----------
            optimizer_selector: BaseSelector
                Initialised BaseSelector object which specifies how optimizers are selected and initialised. See
                glompo.opt_selectors.BaseSelector for detailed documentation.
            manager_kwargs: Optional[Dict[str, Any]] = None
                A dictionary of optional arguments to the GloMPOManager initialisation function.
                Notes that all arguments are accepted but required GloMPO arguments 'task', 'n_parms', 'bounds' and
                'max_jobs' will be overwritten as they are passed by the 'minimize' function in accordance with ParAMS
                API.
        """

        if manager_kwargs:
            self.manager_kwargs = manager_kwargs
            for kw in ['task', 'n_parms', 'bounds', 'max_jobs']:
                if kw in self.manager_kwargs:
                    del self.manager_kwargs[kw]
        else:
            self.manager_kwargs = {}

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
        workers: int
            Passed to GloMPO as its 'max_jobs' parameter. The maximum number of optimizers run in parallel.
        callbacks: List[Callable]
            GloMPO ignores the callbacks parameter as it is ambiguous in this context. To control the termination of
            the manager itself use BaseChecker objects passed to GloMPO's convergence_criteria parameter.

            To control individual optimizers with callbacks, pass callback functions through BaseSelector objects to
            GloMPO's optimizer_selector parameter. Callbacks in this sense, however, are discouraged as it defeats
            the purpose of GloMPO's monitored optimization.

            In some cases it may be preferable to pass callback conditions as BaseHunter objects instead.
        """

        warnings.warn("The x0 parameter is ignored by GloMPO. To control the starting locations of optimizers within "
                      "GloMPO make use of its BaseGenerator objects.", RuntimeWarning)
        if callbacks:
            warnings.warn("Callbacks provided to the minimize function are ignored. Callbacks to individual "
                          "optimizers can be passed to GloMPO through BaseSelector objects. Callbacks to control the "
                          "manager itself are passed using GloMPO BaseChecker objects.")

        manager = GloMPOManager(task=function,
                                n_parms=len(x0),
                                bounds=bounds,
                                max_jobs=workers,
                                **self.manager_kwargs)

        result = manager.start_manager()

        return result
