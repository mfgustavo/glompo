

# Native Python imports
import warnings
from time import time
import multiprocessing as mp

# Package imports
from ..generators.basegenerator import BaseGenerator
from ..generators.random import RandomGenerator
from ..convergence.basechecker import BaseChecker
from ..convergence.nkillsafterconv import KillsAfterConvergence
from ..common.namedtuples import *
from ..scope.scope import ParallelOptimizerScope
from ..optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from .gpr import GaussianProcessRegression
from .expkernel import ExpKernel

# Other Python packages
import numpy as np


class GloMPOManager:
    """ Runs given jobs in parallel and tracks their progress using Gaussian Process Regressions.
        Based on these predictions the class will update hyperparameters, kill poor performing jobs and
        intelligently restart others. """

    def __init__(self,
                 task: Callable[[Sequence[float]], float],
                 n_parms: int,
                 optimizers: Dict[str, Union[Type[BaseOptimizer], Tuple[Type[BaseOptimizer],
                                                                        Dict[str, Any], Dict[str, Any]]]],
                 bounds: Sequence[Tuple[float, float]],
                 max_jobs: Optional[int] = None,
                 task_args: Optional[Tuple] = None,
                 task_kwargs: Optional[Dict] = None,
                 convergence_checker: Optional[Type[BaseChecker]] = None,
                 x0_generator: Optional[Type[BaseGenerator]] = None,
                 tmax: Optional[float] = None,
                 fmax: Optional[int] = None,
                 region_stability_check: bool = False,
                 report_statistics: bool = False,
                 history_logging: int = 0,
                 visualisation: bool = False,
                 visualisation_args: Optional[Dict[str, Any]] = None):
        """
        Generates the environment for a globally managed parallel optimization job.

        Parameters
        ----------
        task : Callable[[Sequence[float]], float]
            Function to be minimized. Accepts a 1D sequence of parameter values and returns a single value.
            Note: Must be a standalone function which makes no modifications outside of itself.

        n_parms : int
            The number of parameters to be optimized.

        optimizers : Dict[str, Union[Type[BaseOptimizer], Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]]
            Dictionary of callable optimization functions (which are children of the BaseOptimizer class) with keywords
            describing their behaviour. Recognized keywords are:
                'default': The default optimizer used if any of the below keywords are not set. This is the only
                    optimizer which *must* be set.
                'early': Optimizers which are more global in their search and best suited for early stage optimization.
                'late': Strong local optimizers which are best suited to refining solutions in regions with known good
                    solutions.
                'noisy': Optimizer which should be used in very noisy areas with very steep gradients or discontinuities
            Values can also optionally be (1, 3) tuples of optimizers and dictionaries of optimizer keywords. The first
            is passed to the optimizer during initialisation and the second during execution.
            For example:
                {'default': CMAOptimizer}: The mp_manager will use only CMA-ES type optimizers in all cases and use
                    default values for them. In cases such as this the optimzer should have no non-default parameters
                    other than func, x0 and bounds.
                {'default': CMAOptimizer, 'noisy': (CMAOptimizer, {'sigma': 0.1}, None)}: The mp_manager will act as
                    above but in noisy areas CMAOptimizers are initialised with a smaller step size. No special keywords
                    are passed for the minimization itself.

        bounds : Sequence[Tuple[float, float]]
            Sequence of tuples of the form (min, max) limiting the range of each parameter.

        max_jobs : int
            The maximum number of local optimizers run in parallel at one time. Defaults to one less than the number of
            CPUs available to the system with a minimum of 1

        task_args : Optional[Tuple]]
            Optional arguments passed to task with every call.

        task_kwargs : Optional[Dict]
            Optional keyword arguments passed to task with every call.

        convergence_checker: Optional[Type[BaseChecker]]
            Criteria used for convergence. Supported arguments:
                'n_kills_after_conv':  The mp_manager delivers the best answer obtained after n optimizers have been
                    killed after at least one has been allowed to converge. Replace n with an integer value. The default
                    value is 0_kills_after_conv which delivers the answer produced as soon as an optimizer converges.

        x0_generator : Optional[Type[BaseGenerator]]
            An instance of a subclass of BaseGenerator which produces starting points for the optimizer. If not provided
            a random generator is used.

        tmax : Optional[int]
            Maximum number of seconds the optimizer is allowed to run before exiting (gracefully) and delivering the
            best result seen up to this point.
            Note: If tmax is reached GloMPO will not run region_stability_checks.

        fmax : Optional[int]
            Maximum number of function calls that are allowed between all optimizers.

        region_stability_check: bool = False
            If True, local optimizers are started around a candidate solution which has been selected as a final
            solution. This is used to measure its reproducibility. If the check fails, the mp_manager resumes looking
            for another solution.

        report_statistics: bool = False
            If True, the mp_manager reports the statistical significance of the suggested solution.

        history_logging: int = 0
            Indicates the level of logging the user would like:
                0 - No log files are saved.
                1 - The log file of the optimizer from which the final solution was extracted is saved.
                2 - The log file of every started optimizer is saved.
                3 - The log file and screen output of every optimizer is saved.

        visualisation : bool
            If True then a dynamic plot is generated to demonstrate the performance of the optimizers. Further options
            (see visualisation_args) allow this plotting to be recorded and saved as a film.

        visualisation_args : Union[dict, None]
            Optional arguments to parameterize the dynamic plotting feature. See ParallelOptimizationScope.
        """

        # Signal dictionary
        self._SIGNAL_DICT = {0: "I've made some nan's"}

        # Save and wrap task
        def task_args_wrapper(func, *args, **kwargs):
            def wrapper(x):
                return func(x, *args, **kwargs)
            return wrapper

        if not callable(task):
            raise TypeError(f"{task} is not callable.")
        self.task = task_args_wrapper(task, task_args, task_kwargs)

        # Save n_parms
        if n_parms > 0 and isinstance(n_parms, int):
            self.n_parms = n_parms
        else:
            raise ValueError(f"Cannot parse n_parms = {n_parms}. Only positive integers are allowed.")

        # Save optimizers
        if 'default' not in optimizers:
            raise ValueError("'default' not found in optimizer dictionary. This value must be set.")
        for key in optimizers:
            if not isinstance(optimizers[key], tuple):
                optimizers[key] = (optimizers[key], None, None)
            if not issubclass(optimizers[key][0], BaseOptimizer):
                raise TypeError(f"{optimizers[key][0]} not an instance of BaseOptimizer.")
        self.optimizers = optimizers

        # Save bounds
        self.bounds = []
        if len(bounds) != n_parms:
            raise ValueError(f"Number of parameters (n_parms) and number of bounds are not equal")
        for bnd in bounds:
            if bnd[0] == bnd[1]:
                raise ValueError(f"Bounds min and max cannot be equal. Rather fix its value and remove it from the "
                                 f"list of parameters. Fixed values can be supplied through task_args or task_kwargs.")
            if bnd[1] < bnd[0]:
                raise ValueError(f"Bound min cannot be larger than max.")
            self.bounds.append(Bound(bnd[0], bnd[1]))

        # Save max_jobs
        if isinstance(max_jobs, int):
            if max_jobs > 0:
                self.max_jobs = max_jobs
            else:
                raise ValueError(f"Cannot parse max_jobs = {max_jobs}. Only positive integers are allowed.")
        else:
            raise TypeError(f"Cannot parse max_jobs = {max_jobs}. Only positive integers are allowed.")

        # Save convergence criteria
        if convergence_checker:
            if isinstance(convergence_checker, BaseChecker):
                self.convergence_checker = convergence_checker
            else:
                raise TypeError(f"convergence_checker not an instance of a subclass of BaseChecker.")
        else:
            self.convergence_checker = KillsAfterConvergence()

        # Save x0 generator
        if x0_generator:
            if isinstance(x0_generator, BaseGenerator):
                self.x0_generator = x0_generator
            else:
                raise TypeError(f"x0_generator not an instance of a subclass of BaseOptimizer.")
        else:
            self.x0_generator = RandomGenerator(self.bounds)

        # Save max conditions and counters
        self.tmax = np.clip(int(tmax), 1, None) if tmax or tmax == 0 else np.inf
        self.fmax = np.clip(int(fmax), 1, None) if fmax or fmax == 0 else np.inf

        self.t_start = None
        self.o_counter = 0
        self.f_counter = 0
        self.conv_counter = 0  # Number of converged optimizers
        self.kill_counter = 0  # Number of killed jobs
        self.hunt_counter = 0  # Number of hunting jobs started
        self.hyop_counter = 0  # Number of hyperparameter optimization jobs started
        self.hunt_victims = {}  # opt_ids of killed jobs and timestamps when the signal was sent

        # Save behavioural args
        self.region_stability_check = bool(region_stability_check)
        self.report_statistics = bool(report_statistics)
        self.history_logging = np.clip(int(history_logging), 0, 3)
        self.visualisation = visualisation
        if visualisation:
            self.scope = ParallelOptimizerScope(num_streams=max_jobs, **visualisation_args)

        # Setup multiprocessing variables
        self.optimizer_processes = {}
        self.hyperparm_processes = {}
        self.hunting_processes = {}
        self.signal_pipes = {}

        self.mp_manager = mp.Manager()
        self.optimizer_queue = self.mp_manager.Queue()
        self.hyperparm_queue = self.mp_manager.Queue()
        self.hunting_queue = self.mp_manager.Queue()

        # Setup GPRs
        self.gprs = {}

    def start_manager(self) -> MinimizeResult:
        """ Begins the optimization routine.

        Parameters
        ----------
        """
        self.t_start = time()

        for i in range(self.max_jobs):
            opt = self._setup_new_optimizer()
            self._start_new_job(*opt)

        converged = self.convergence_checker.converged(self)

        while self.tmax < time() - self.t_start and self.f_counter < self.fmax and not converged:

            # Check hunt_queue

            if not self.hunting_queue.empty():
                hunt = self.hunting_queue.get()
                self._shutdown_job(hunt.victim)

            #   Force kill any stragglers
            for id in self.hunt_victims:
                if time() - self.hunt_victims[id] > 20 * 60:
                    self._force_shutdown_job(id)

            #   Is there a space in the queue? Start new job in its place
            count = sum([int(proc.is_alive()) for proc in self.optimizer_processes])
            if count < self.max_jobs:
                opt = self._setup_new_optimizer()
                self._start_new_job(*opt)

            # Check results_queue

            #   When taking a point make sure the process is still alive, might be the last point

            #   If results are from a killed optimizer, ignore them

            #   Send results to x0_generator

            #   Send results to GPRs

            #   Send results to scope

            # Check hyperparm queue

            #   Update hyperparms

            #   Ensure hyperparms aren't stupid superwide solution

            # Poke processes (especially ones we havent heard from in a while)

            # Every n iterations (Careful, 'n' is different for each optimizer

            #   Start hyperparams job

            #   Start hunting jobs

            # Update counters

            pass

        # Check answer

        # End gracefully

        #   Make movie

        #   Join all processes

        #   Delete temp files

        #   Including crash files that are not otherwise cleaned up

        # Possibly check answer here (This could lead to a new loop and definitely a new end gracefully)

    def _start_new_job(self, opt_id: int, optimizer: BaseOptimizer, call_kwargs):
        task = self.task
        x0 = self.x0_generator.generate()
        bounds = np.array(self.bounds)

        self.optimizer_processes[opt_id] = mp.Process(target=optimizer.minimize,
                                                      args=(task, x0, bounds),
                                                      kwargs=call_kwargs,
                                                      daemon=True)
        self.optimizer_processes[opt_id].start()
        if self.visualisation:
            if opt_id not in self.scope.streams:
                self.scope.add_stream(opt_id)

    def _shutdown_job(self, opt_id):
        # TODO The signal to end will not be read instantly unless listeners are added to the wrappers
        #  alternatively we can send the signal and if it doesnt co-operate in x time we force it to die.
        #  Optimal solution would be to have both, first solution is 'proper' and the second is an extra layer of
        #  safety.

        self.hunt_victims[opt_id] = time()

        self.signal_pipes[opt_id].send((1, None))

        try:
            del self.gprs[opt_id]
        except KeyError:
            warnings.warn("Repeated attempts to shutdown single optimizer. ", UserWarning)

        if self.visualisation:
            self.scope.update_kill(opt_id)

    def _force_shutdown_job(self, opt_id):
        """ Warning there is a chance that this will corrupt the results_queue. Only use this in circumstances where you
            are sure the process is frozen.
        """
        if opt_id in self.optimizer_processes:
            self.optimizer_processes[opt_id].terminate()

    def _start_hunt(self):
        pass

    # TODO Check status/ check health. If we haven't heard from an optimizer in a while we need to make sure the thing
    #  is still running properly. Maybe we need listeners here to detect when a process ends.

    def _optimize_hyperparameters(self):
        pass

    def _explore_basin(self):
        pass

    def _setup_new_optimizer(self) -> OptimizerPackage:
        self.o_counter += 1

        # TODO add intelligence to pick optimizer?
        selected, init_kwargs, call_kwargs = self.optimizers['default']

        self.gprs[self.o_counter] = GaussianProcessRegression(kernel=ExpKernel(alpha=0.100,
                                                                               beta=5.00),
                                                              dims=1)

        self.signal_pipes[self.o_counter], child_pipe = mp.Pipe()

        optimizer = selected(**init_kwargs, signal_pipe=child_pipe)

        return OptimizerPackage(self.o_counter, optimizer, call_kwargs)
