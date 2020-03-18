

# Native Python imports
from typing import *

# External package imports
import multiprocessing as mp

# AMS imports
from scm.params.optimizers.base import *
from scm.params.core.parameteroptimization import Optimization

# Package imports
from core.scope import GloMPOScope


__all__ = ("ParamsGlompoOptimizer",)


class ParamsGlompoOptimizer(BaseOptimizer):
    """ Runs given jobs in parallel and tracks their progress using Gaussian Process Regressions.
        Based on these predictions the class will update hyperparameters, kill poor performing jobs and
        intelligently restart others. """

    needscaler = False  # TODO: Explore this??? Depends on the individual jobs run

    def __init__(self, optimizers: dict, max_jobs: int, visualisation: bool = False,
                 visualisation_args: Union[dict, None] = None):
        """
        Generates the environment for a globally managed parallel optimization job.

        Parameters
        ----------
        optimizers: dict, Callables
            Dictionary of callable optimization functions with keywords describing their behaviour. Recognized keywords
            are:
                'default': The default optimizer used if any of the below keywords are not set. This is the only
                    optimizer which *must* be set.
                'early': Optimizers which are more global in their search and best suited for early stage optimization.
                'late': Strong local optimizers which are best suited to refining solutions in regions with known good
                    solutions.
                'noisy': Optimizer which should be used in very noisy areas with very steep gradients or discontinuities

        max_jobs : int
            The maximum number of local optimizers run in parallel.
        visualisation : bool
            If True then a dynamic plot is generated to demonstrate the performance of the optimizers. Further options
            (see visualisation_args) allow this plotting to be recorded and saved as a film.
        visualisation_args : Union[dict, None]
            Optional arguments to parameterize the dynamic plotting feature.
        """

        self.result = MinimizeResult()

        if 'default' not in optimizers:
            raise ValueError("'default' not found in optimizer dictionary. This value must be set.")
        elif not any([callable(opt.minimize) for opt in optimizers.values()]):
            raise ValueError("Incompatible optimizer found in dictionary. Ensure it is a child of the BaseOptimizer "
                             "class and has a minimize() method.")
        else:
            self.optimizers = optimizers

        self.max_jobs = max_jobs

        self.optimizer_jobs = []
        self.hyperparm_jobs = []
        self.hunting_jobs = []

        self.manager = mp.Manager()
        self.optimizer_queue = self.manager.Queue()
        self.hyperparm_queue = self.manager.Queue()
        self.hunting_queue = self.manager.Queue()

        if visualisation:
            self.scope = GloMPOScope(num_streams=max_jobs, **visualisation_args)

    # noinspection PyMethodOverriding
    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, parent: Union[Type[Optimization], None] = None) -> MinimizeResult:
        """

        Parameters
        ----------
        function : Optimization.run
        x0 : Sequence[float]
        bounds : Sequence[Tuple[float, float]]
        callbacks : Callable
            Callbacks used to terminate optimizers early. Their use is *strongly* discouraged when using GloMPOManager
        parent : Union[Type[Optimization], None]
            Instance of the Optimization class which bundles function together. Allows access to various optimization
            functions like individual error function contributions or performance at each iteration.

        Returns
        -------
        MinimizeResult
            The result of the optimization. Can be accessed by:
                success : bool
                    Whether the optimization was successful or not
                x : numpy.array
                    The optimized parameters
                fx : float
                    The corresponding |Fitfunc| value of `x`
        """
        print(function._callbacks)

    # noinspection PyMethodOverriding
    def callstop(self, reason=None):
        pass

    def _start_new_job(self):
        pass

    def _kill_job(self):
        pass

    def _start_hunt(self):
        pass

    def _optimize_hyperparameters(self):
        pass

    def _explore_basin(self):
        pass
