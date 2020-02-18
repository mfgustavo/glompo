

# Native Python imports
from typing import *
import multiprocessing as mp

# Package imports
from scope.scope import ParallelOptimizerScope
from optimizers.baseoptimizer import BaseOptimizer


class Result(NamedTuple):
    x: Sequence[float]
    fx: float
    stats: Dict[str, Any]
    origin: Dict[str, Any]  # Optimizer name, settings, starting point and termination condition


class GloMPOOptimizer:
    """ Runs given jobs in parallel and tracks their progress using Gaussian Process Regressions.
        Based on these predictions the class will update hyperparameters, kill poor performing jobs and
        intelligently restart others. """

    def __init__(self,
                 task: Callable[Sequence[float], float],
                 optimizers: Dict[str, Callable[Sequence[Any],]],
                 bounds: Sequence[Tuple[float, float]],
                 max_jobs: int,

                 convergence_criteria: str,
                 x0_criteria: str,
                 tmax: Optional[float] = None,
                 omax: Optional[int] = None,



                 region_stability_check: bool = False,
                 report_statistics: bool = False,
                 history_logging: int = 0,


                 visualisation: bool = False,
                 visualisation_args: Optional[Dict[str, Any]] = None):
        """
        Generates the environment for a globally managed parallel optimization job.

        Parameters
        ----------
        task: Callable[Sequence[float], float]
            Function to be minimized. Accepts a 1D sequence of parameter values and returns a single value.
            Note: Must be a standalone function which makes no modifications outside of itself.
        optimizers: dict, Callables
            Dictionary of callable optimization functions (which are children of the BaseOptimizer class) with keywords
            describing their behaviour. Recognized keywords are:
                'default': The default optimizer used if any of the below keywords are not set. This is the only
                    optimizer which *must* be set.
                'early': Optimizers which are more global in their search and best suited for early stage optimization.
                'late': Strong local optimizers which are best suited to refining solutions in regions with known good
                    solutions.
                'noisy': Optimizer which should be used in very noisy areas with very steep gradients or discontinuities
        bounds: Sequence[Tuple[float, float]]
            Sequence of tuples of the form (min, max) limiting the range of each parameter.


        max_jobs : int
            The maximum number of local optimizers run in parallel.
        visualisation : bool
            If True then a dynamic plot is generated to demonstrate the performance of the optimizers. Further options
            (see visualisation_args) allow this plotting to be recorded and saved as a film.
        visualisation_args : Union[dict, None]
            Optional arguments to parameterize the dynamic plotting feature.

        convergence_criteria: str
            Criteria used for convergence. Supported arguments:
                'omax': The manager delivers the best answer obtained after initialising omax optimizers. Note that
                    these are not guaranteed to be allowed to terminate themselves i.e. GloMPO may still kill them
                    early.
                'sing_conv': The manager delivers the answer obtained by the first optimizer to be allowed to reach full
                    convergence.
                'n_conv':  The manager delivers the answer obtained by the nth optimizer to be allowed to reach full
                    convergence. Replace n with an integer value.
                    WARNING: It is strongly recommended that this criteria is used with some early termination criteria
                        such as omax or tmax since later optimizations may not improve on earlier runs and thus may not
                        be allowed to converge, hence this criteria is never reached.
        x0_criteria: str
            Criteria used for selection of the next starting point. Supported arguments:
                'gp': A Guassian process regression is used to estimate areas with low error. Requires many function
                    evaluations (i.e many optimizers started) to begin producing effective suggestions.
                'es': Uses evolutionary strategies of crossover and mutation to generate new starting sets. More
                    reliable and cheaper but does not learn the behaviour of the error function.
                'es+gp': Uses 'es' for the early part of the optimization and switches to 'gp' in the late stage after
                    sufficient information has be acquired.
                'rand': New starting points are selected entirely randomly.
        tmax: Optional[float]
        omax: Optional[int]
        region_stability_check: bool = False
        report_statistics: bool = False
        history_logging: int = 0
        """

        if 'default' not in optimizers:
            raise ValueError("'default' not found in optimizer dictionary. This value must be set.")
        for optimizer in optimizers.values():
            if not isinstance(optimizer, BaseOptimizer):
                raise TypeError(f"{optimizer} not an instance of BaseOptimizer.")

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
            self.scope = ParallelOptimizerScope(num_streams=max_jobs, **visualisation_args)

    def start_manager(self) -> Result:
        """ Begins the optimization routine.

        Parameters
        ----------
        """

        pass

    # TODO Selection of new starting points can be driven by a Gaussian Process (Probably unrealistic unless a LOT of
    #  optimizers are started), Genetic Algorithms (Reasonable but still no feeling for the error surface) or random
    #  (very easy to implement and possible same quality as the others given the sparse space).
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
