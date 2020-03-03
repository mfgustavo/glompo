

# Native Python imports
import warnings
from datetime import datetime
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
from .logger import Logger

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
                 visualisation_args: Optional[Dict[str, Any]] = None,
                 allow_forced_terminations: bool = True,
                 verbose: bool = True):
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

        visualisation_args : Optional[Dict[str, Any]]
            Optional arguments to parameterize the dynamic plotting feature. See ParallelOptimizationScope.

        allow_forced_terminations : bool
            If True GloMPO will force terminate processes that do not respond to normal kill signals in an appropriate
            amount of time. This runs the risk of corrupting the results queue but ensures resources are not wasted on
            hanging processes.

        verbose : bool
            If True, GloMPO will print progress messages during the optimization otherwise it will be silent.
        """
        self.verbose = verbose
        self._optional_print("Initializing Manager ... ")

        # Signal dictionary
        self._SIGNAL_DICT = {0: "Normal Termination (Args have reason)",
                             1: "I have made some nan's... oopsie... >.<"}

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
        # TODO Update counters everywhere
        self.conv_counter = 0  # Number of converged optimizers
        self.kill_counter = 0  # Number of killed jobs
        self.hunt_counter = 0  # Number of hunting jobs started
        self.hyop_counter = 0  # Number of hyperparameter optimization jobs started
        self.hunt_victims = {}  # opt_ids of killed jobs and timestamps when the signal was sent

        # Save behavioural args
        self.allow_forced_terminations = allow_forced_terminations
        self.region_stability_check = bool(region_stability_check)
        self.report_statistics = bool(report_statistics)
        self.history_logging = np.clip(int(history_logging), 0, 3)
        self.visualisation = visualisation
        # TODO: Put scope into a seperate process so glompo performance is not effected
        if visualisation:
            self.scope = ParallelOptimizerScope(**visualisation_args)

        # Setup multiprocessing variables
        self.optimizer_processes = {}
        self.hyperparm_processes = {}
        self.hunting_processes = {}
        self.signal_pipes = {}
        self.pause_events = {}

        self.mp_manager = mp.Manager()
        self.optimizer_queue = self.mp_manager.Queue()
        self.hyperparm_queue = self.mp_manager.Queue()
        self.hunting_queue = self.mp_manager.Queue()

        # Setup GPRs
        self.gprs = {}
        self.log = Logger()

        self._optional_print("Initialization Done")

    def start_manager(self) -> MinimizeResult:
        """ Begins the optimization routine.

        Parameters
        ----------
        """

        self._optional_print("------------------------------------\n"
                             "Starting GloMPO Optimization Routine\n"
                             "------------------------------------\n")

        self.t_start = time()

        for i in range(self.max_jobs):
            opt = self._setup_new_optimizer()
            self._start_new_job(*opt)

        converged = self.convergence_checker.converged(self)
        if converged:
            self._optional_print("!!! Convergence Reached !!!")

        while self.tmax < time() - self.t_start and self.f_counter < self.fmax and not converged:

            # Check signals
            self._optional_print("Checking optimizer signals...")
            for opt_id in self.signal_pipes:
                pipe = self.signal_pipes[opt_id]
                if pipe.poll():
                    key, message = pipe.recv()
                    self._optional_print(f"Signal {key} from {opt_id}.")
                    if key == 0:
                        self.log.put_metadata(opt_id, "Stop Time", datetime.now())
                        self.log.put_metadata(opt_id, "End Condition", message)
                        # TODO More, if a optimizer is done we need to cleanup after it (remove from gprs) for example
                    elif key == 1:
                        # TODO Deal with 1 signals
                        pass
                    elif key == 9:
                        self.log.put_message(opt_id, message)
                else:
                    self._optional_print(f"No signals from {opt_id}.")
            self._optional_print(f"Signal check done.")

            # Check hunt_queue
            self._optional_print("Checking hunt results...")
            if not self.hunting_queue.empty():
                hunt = self.hunting_queue.get()
                self.hunting_processes[hunt.hunt_id].join()
                self._optional_print(f"Manager wants to kill {hunt.victim}")
                self._shutdown_job(hunt.victim)
            else:
                self._optional_print("No hunts successful.")
            self._optional_print("Hunt check done.")

            # Force kill any stragglers
            if self.allow_forced_terminations:
                for id_num in self.hunt_victims:
                    # TODO: This check fails in the case where optimizers are paused and then restarted
                    if time() - self.hunt_victims[id_num] > 20 * 60 and self.optimizer_processes[id_num].is_alive():
                        self._force_shutdown_job(id_num)

            # Check results_queue
            self._optional_print("Checking optimizer iteration results...")
            if not self.optimizer_queue.empty():
                res = self.optimizer_queue.get()
                if not any([res.opt_id == victim for victim in self.hunt_victims]):

                    self.x0_generator.update(res.x, res.fx)
                    self.log.put_iteration(res.opt_id, res.n_iter, res.x, res.fx)

                    # Send results to GPRs
                    trained = False
                    # TODO Intelligent GPR training. Take n well dispersed points
                    if res.n_iter % 10 == 0:
                        trained = True
                        self._optional_print(f"Result from {res.opt_id} sent to GPR")
                        self.gprs[res.opt_id].add_known(res.x, res.fx)

                        mean = np.mean(self.log.get_history(res.opt_id, "fx"))
                        sigma = np.std(self.log.get_history(res.opt_id, "fx"))

                        self.gprs[res.opt_id].rescale((mean, sigma))

                    if self.visualisation:
                        self.scope.update_optimizer(res.opt_id, (res.x, res.fx))
                        if trained:
                            self.scope.update_scatter(res.opt_id, (res.x, res.fx))
                        if res.final:
                            self.scope.update_norm_terminate(res.opt_id)
            else:
                self._optional_print("No results found.")
            self._optional_print("Iteration results check done.")

            # Start new processes if possible
            self._optional_print("Checking for available optimizer slots...")
            count = sum([int(proc.is_alive()) for proc in self.optimizer_processes])
            # NOTE: is_alive joins any dead processes
            while count < self.max_jobs:
                opt = self._setup_new_optimizer()
                self._start_new_job(*opt)
                count += 1
            self._optional_print("New optimizer check done.")

            # Check hyperparm queue
            # TODO Check quality of new hyperparms to stop superwide
            # TODO Join hunt and hyperparm processes
            self._optional_print("Checking for converged hyperparameter optimizations...")
            if not self.hyperparm_queue.empty():
                res = self.hyperparm_queue.get_nowait()
                self._optional_print(f"New hyperparameters found for {res.opt_id}")

                self.gprs[res.opt_id].kernel.alpha = res.alpha
                self.gprs[res.opt_id].kernel.beta = res.beta
                self.gprs[res.opt_id].sigma_noise = res.sigma

                if self.visualisation:
                    self.scope.update_opt_end(res.opt_id)
            else:
                self._optional_print("No results found.")
            self._optional_print("New hyperparameter check done.")

            # Poke processes (especially ones we havent heard from in a while)

            # Every n iterations (Careful, 'n' is different for each optimizer

            #   Start hyperparams job

            #   Start hunting jobs

            # Update counters

            pass
        self._optional_print("Exiting manager loop.")

        # Check answer

        # End gracefully

        #   Make movie

        #   Join all processes

        #   Delete temp files

        #   Including crash files that are not otherwise cleaned up

        #   Save log

        # Possibly check answer here (This could lead to a new loop and definitely a new end gracefully)
        self._optional_print("-----------------------------------\n"
                             "GloMPO Optimization Routine... DONE\n"
                             "-----------------------------------\n")

    def _start_new_job(self, opt_id: int, optimizer: BaseOptimizer, call_kwargs):

        self._optional_print(f"Starting Optimizer: {opt_id}")

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

        # TODO Shutdown other things like pipes, events, gprs, what is allowed to be left in optimizer_process dict

        self.hunt_victims[opt_id] = time()

        self.signal_pipes[opt_id].send((1, None))
        self._optional_print(f"Termination signal sent to {opt_id}")

        self.log.put_metadata(opt_id, "Stop Time", datetime.now())
        self.log.put_metadata(opt_id, "End Condition", "GloMPO Termination")

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

        # TODO Shutdown other things like pipes, events, gprs, what is allowed to be left in optimizer_process dict

        if opt_id in self.optimizer_processes:
            self.optimizer_processes[opt_id].terminate()
            warnings.warn(f"Forced termination signal sent to optimizer {opt_id}.", UserWarning)

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

        self._optional_print(f"Selected Optimizer:\n"
                             f"\tOptimizer ID: {self.o_counter}\n"
                             f"\tType: {type(selected).__name__}")

        self.gprs[self.o_counter] = GaussianProcessRegression(kernel=ExpKernel(alpha=0.100,
                                                                               beta=5.00),
                                                              dims=1)

        self.signal_pipes[self.o_counter], child_pipe = mp.Pipe()

        event = self.mp_manager.Event()
        event.set()
        self.pause_events[self.o_counter] = event

        optimizer = selected(**init_kwargs,
                             opt_id=self.o_counter,
                             signal_pipe=child_pipe,
                             results_queue=self.optimizer_queue,
                             pause_flag=event)

        self.log.add_optimizer(self.o_counter, type(optimizer).__name__, datetime.now())

        return OptimizerPackage(self.o_counter, optimizer, call_kwargs)

    def _optional_print(self, message: str):
        if self.verbose:
            print(message)
