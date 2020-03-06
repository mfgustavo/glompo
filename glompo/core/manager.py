

# Native Python imports
import shutil
import warnings
from datetime import datetime
from functools import wraps
from time import time
import multiprocessing as mp
import traceback
import os

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
import yaml


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
                1 - Only the manager log is saved.
                2 - The manager log and log file of the optimizer from which the final solution was extracted is saved.
                3 - The manager log and the log files of every started optimizer is saved.

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

        # CONSTANTS
        self._SIGNAL_DICT = {0: "Normal Termination (Args have reason)",
                             1: "I have made some nan's... oopsie... >.<"}
        self._TOO_LONG = 20 * 60

        # Save and wrap task
        def task_args_wrapper(func, args, kwargs):
            @wraps(func)
            def wrapper(x):
                return func(x, *args, **kwargs)
            return wrapper

        if not callable(task):
            raise TypeError(f"{task} is not callable.")
        if not task_args:
            task_args = ()
        if not task_kwargs:
            task_kwargs = {}
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
                optimizers[key] = (optimizers[key], {}, {})
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
        self.dt_start = None
        self.o_counter = 0
        self.f_counter = 0
        # TODO Update counters everywhere
        self.conv_counter = 0  # Number of converged optimizers
        self.hunt_counter = 0  # Number of hunting jobs started
        self.hyop_counter = 0  # Number of hyperparameter optimization jobs started
        self.hunt_victims = {}  # opt_ids of killed jobs and timestamps when the signal was sent

        # Save behavioural args
        self.allow_forced_terminations = allow_forced_terminations
        self.region_stability_check = bool(region_stability_check)
        self.report_statistics = bool(report_statistics)
        self.history_logging = np.clip(int(history_logging), 0, 3)
        self.log = Logger()
        self.visualisation = visualisation
        # TODO: Put scope into a seperate process so glompo performance is not effected
        if visualisation:
            self.scope = ParallelOptimizerScope(**visualisation_args)

        # Setup multiprocessing variables
        self.optimizer_packs = {}  # Dict[opt_id (int): ProcessPackage (NamedTuple)]
        self.hyperparm_processes = {}
        self.hunting_processes = {}
        self.graveyard = set()  # opt_ids of known non-active optimizers
        self.last_feedback = {}

        self.mp_manager = mp.Manager()
        self.optimizer_queue = self.mp_manager.Queue()
        self.hyperparm_queue = self.mp_manager.Queue()
        self.hunting_queue = self.mp_manager.Queue()

        # Purge Old Results
        files = os.listdir(".")
        # TODO Make more user friendly after debugging
        if any([file in files for file in ["glompo_manager_log.yml", "glompo_optimizer_logs",
                                           "glompo_best_optimizer_log"]]):
            print("Old results found. Deleting...")
            shutil.rmtree("glompo_manager_log.yml", ignore_errors=True)
            shutil.rmtree("glompo_optimizer_logs", ignore_errors=True)
            shutil.rmtree("glompo_best_optimizer_log", ignore_errors=True)
            self._optional_print("\tDeleted old results.")

        self._optional_print("Initialization Done")

    def start_manager(self) -> MinimizeResult:
        """ Begins the optimization routine. """

        best_id = 0
        result = Result(None, None, None, None)
        converged = False
        reason = ""
        caught_exception = False

        try:
            self._optional_print("------------------------------------\n"
                                 "Starting GloMPO Optimization Routine\n"
                                 "------------------------------------\n")

            self.t_start = time()
            self.dt_start = datetime.now()

            for i in range(self.max_jobs):
                opt = self._setup_new_optimizer()
                self._start_new_job(*opt)

            while time() - self.t_start < self.tmax and self.f_counter < self.fmax and not converged:

                # Start new processes if possible
                self._optional_print("Checking for available optimizer slots...")
                processes = [pack.process for pack in self.optimizer_packs.values()]
                count = sum([int(proc.is_alive()) for proc in processes])
                # NOTE: is_alive joins any dead processes
                while count < self.max_jobs:
                    opt = self._setup_new_optimizer()
                    self._start_new_job(*opt)
                    count += 1
                self._optional_print("New optimizer check done.")

                # Check signals
                self._optional_print("Checking optimizer signals...")
                for opt_id in self.optimizer_packs:
                    self._check_signals(opt_id)
                self._optional_print(f"Signal check done.")

                # Check hunt_queue
                self._optional_print("Checking hunt results...")
                if not self.hunting_queue.empty():
                    while not self.hunting_queue.empty():
                        hunt = self.hunting_queue.get()
                        self.hunting_processes[hunt.hunt_id].join()
                        del self.hunting_processes[hunt.hunt_id]
                        for victim in hunt.victims:
                            self._optional_print(f"\tManager wants to kill {victim}")
                            self._shutdown_job(victim)
                else:
                    self._optional_print("\tNo hunts successful.")
                self._optional_print("Hunt check done.")

                # Force kill any stragglers
                if self.allow_forced_terminations:
                    for id_num in self.hunt_victims:
                        # TODO: This check fails in the case where optimizers are paused and then restarted
                        if time() - self.hunt_victims[id_num] > self._TOO_LONG and \
                                self.optimizer_packs[id_num].process.is_alive():
                            self.optimizer_packs[id_num].process.terminate()
                            self.log.put_message(id_num, "Force terminated due to no feedback after kill signal "
                                                         "timeout.")
                            self.log.put_metadata(id_num, "Approximate Stop Time", datetime.now())
                            self.log.put_metadata(id_num, "End Condition", "Forced GloMPO Termination")
                            warnings.warn(f"Forced termination signal sent to optimizer {id_num}.", UserWarning)

                # Check results_queue
                self._optional_print("Checking optimizer iteration results...")
                i_count = 0
                while not self.optimizer_queue.empty() and i_count < 10:
                    res = self.optimizer_queue.get()
                    # TODO Remove line
                    print(f"\tResult {res.n_iter} from {res.opt_id}")
                    i_count += 1
                    self.last_feedback[res.opt_id] = time()
                    self.f_counter += res.n_icalls
                    if self.f_counter > self.fmax:
                        break
                    if not any([res.opt_id == victim for victim in self.hunt_victims]):

                        self.x0_generator.update(res.x, res.fx)
                        self.log.put_iteration(res.opt_id, res.n_iter, list(res.x), res.fx)
                        self._optional_print(f"Result from {res.opt_id} @ iter {res.n_iter} fx = {res.fx}")

                        # Send results to GPRs
                        trained = False
                        # TODO Intelligent GPR training. Take n well dispersed points
                        # TODO Better decisions on when to start hyper parm jobs
                        if res.n_iter % 10 == 0:
                            trained = True
                            self._optional_print(f"\tResult from {res.opt_id} sent to GPR")
                            self.optimizer_packs[res.opt_id].gpr.add_known(res.n_iter, res.fx)

                            # Start new hyperparameter optimization job
                            # TODO Reallow hyperparam jobs
                            # if len(self.optimizer_packs[res.opt_id].gpr.training_values()) % 3 == 0:
                            #     self._start_hyperparam_job(res.opt_id)

                            mean = np.mean(self.log.get_history(res.opt_id, "fx"))
                            sigma = np.std(self.log.get_history(res.opt_id, "fx"))

                            self.optimizer_packs[res.opt_id].gpr.rescale((mean, sigma))

                        if self.visualisation:
                            self.scope.update_optimizer(res.opt_id, (res.n_iter, res.fx))
                            if trained:
                                self.scope.update_scatter(res.opt_id, (res.n_iter, res.fx))

                                # TODO Check how the gpr is drawn. Too coarse? loose? What about when it grow?
                                i_max = len(self.log.get_history(res.opt_id, "fx"))
                                if self.scope.visualise_gpr:
                                    i_range = np.linspace(0, i_max, 200)
                                    mu, sigma = self.optimizer_packs[res.opt_id].gpr.sample_all(i_range)
                                    self.scope.update_gpr(res.opt_id, i_range, mu, mu - 2*sigma, mu + 2*sigma)
                                else:
                                    mu, sigma = self.optimizer_packs[res.opt_id].gpr.sample_all(i_max+1000)
                                    self.scope.update_mean(res.opt_id, mu, sigma)
                            if res.final:
                                self.scope.update_norm_terminate(res.opt_id)
                else:
                    self._optional_print("\tNo results found.")
                self._optional_print("Iteration results check done.")

                # Check processes' statuses
                # TODO Problem check signal first
                for opt_id in self.optimizer_packs:
                    if opt_id not in self.graveyard and not self.optimizer_packs[opt_id].process.is_alive():
                        exitcode = self.optimizer_packs[opt_id].process.exitcode
                        self.graveyard.add(opt_id)
                        if exitcode == 0:
                            self._check_signals(opt_id)
                            self.conv_counter += 1
                            if opt_id not in self.graveyard:
                                self.log.put_message(opt_id, "Terminated normally without sending a minimization "
                                                             "complete signal to the manager.")
                                warnings.warn(f"Optimizer {opt_id} terminated normally without sending a minimization "
                                              f"complete signal to the manager.", UserWarning)
                                self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                                self.log.put_metadata(opt_id, "End Condition", "Normal termination (Reason unknown)")
                        else:
                            self.log.put_message(opt_id, f"Terminated in error with code {-exitcode}")
                            warnings.warn(f"Optimizer {opt_id} terminated in error with code {-exitcode}", UserWarning)
                            self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                            self.log.put_metadata(opt_id, "End Condition", f"Error termination (exitcode {-exitcode}).")
                    if self.optimizer_packs[opt_id].process.is_alive() and time() - self.last_feedback[opt_id] > \
                            self._TOO_LONG and self.allow_forced_terminations:
                        warnings.warn(f"Optimizer {opt_id} seems to be hanging. Forcing termination.", UserWarning)
                        self.log.put_message(opt_id, "Force terminated due to no feedback timeout.")
                        self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                        self.log.put_metadata(opt_id, "End Condition", "Forced GloMPO Termination")
                        self.optimizer_packs[opt_id].process.terminate()

                # Check hyperparm queue
                # TODO Check quality of new hyperparms to stop superwide
                self._optional_print("Checking for converged hyperparameter optimizations...")
                if not self.hyperparm_queue.empty():
                    while not self.hyperparm_queue.empty():
                        res = self.hyperparm_queue.get_nowait()
                        self.hyperparm_processes[res.hyper_id].join()
                        del self.hyperparm_processes[res.hyper_id]

                        self._optional_print(f"\tNew hyperparameters found for {res.opt_id}")

                        self.optimizer_packs[res.opt_id].gpr.kernel.alpha = res.alpha
                        self.optimizer_packs[res.opt_id].gpr.kernel.beta = res.beta
                        self.optimizer_packs[res.opt_id].gpr.sigma_noise = res.sigma

                        if self.visualisation:
                            self.scope.update_opt_end(res.opt_id)
                else:
                    self._optional_print("\tNo results found.")
                self._optional_print("New hyperparameter check done.")

                # Start new hunting jobs
                # TODO Better starting criteria than every 50 function calls?
                if self.f_counter % 100 == 0 and self.f_counter > 0:
                    self._start_hunt()

                # Check convergence
                converged = self.convergence_checker.converged(self)
                if converged:
                    self._optional_print("!!! Convergence Reached !!!")

                # Setup candidate solution
                # TODO Improve solution selection
                # TODO Include stats
                # TODO Include origins
                best_fx = np.inf
                best_x = []
                best_stats = None
                best_origin = None
                for opt_id in self.log.storage:
                    history = self.log.get_history(opt_id, "fx_best")
                    if len(history) > 0:
                        opt_best = history[-1]
                        if opt_best < best_fx:
                            best_fx = opt_best
                            best_id = opt_id

                            i = self.log.get_history(opt_id, "i_best")[-1]
                            best_x = self.log.get_history(opt_id, "x")[i-1]
                            best_origin = {"opt_id": opt_id,
                                           "type": self.log.storage[opt_id].metadata["Optimizer Type"]}

                result = Result(best_x, best_fx, best_stats, best_origin)

            self._optional_print("Exiting manager loop.")
            self._optional_print("Exit conditions met: ")
            self._optional_print(f"\tConvergence check: {converged}")
            self._optional_print(f"\ttmax condition: {time() - self.t_start >= self.tmax}")
            self._optional_print(f"\tfmax condition: {self.f_counter >= self.fmax}")

            # Check answer
            # TODO Check answer

            # Join all processes
            for opt_id in self.optimizer_packs:
                if self.optimizer_packs[opt_id].process.is_alive():
                    self.optimizer_packs[opt_id].signal_pipe.send(1)
                    self.graveyard.add(opt_id)
                    self.log.put_metadata(opt_id, "Stop Time", datetime.now())
                    reason = ""
                    reason += "Conv. Crit. " if converged else ""
                    reason += "tmax " if time() - self.t_start >= self.tmax else ""
                    reason += "fmax " if self.f_counter >= self.fmax else ""
                    self.log.put_metadata(opt_id, "End Condition", f"GloMPO Convergence ({reason})")
                    self.optimizer_packs[opt_id].process.join(self._TOO_LONG / 20)

            return result

        except Exception as e:
            caught_exception = True
            warnings.warn(f"Optimization failed. Caught exception: {e}")
            print("".join(traceback.TracebackException.from_exception(e).format()))
        finally:

            # Make movie
            if self.visualisation and self.scope.record_movie:
                self.scope.generate_movie()

            # Save log
            if self.history_logging > 0:
                if caught_exception:
                    reason = "Process Crash"
                with open("glompo_manager_log.yml", "w") as file:
                    optimizers = {}
                    for name in self.optimizers:
                        optimizers[name] = {"Class": self.optimizers[name][0].__name__,
                                            "Init Args": self.optimizers[name][1],
                                            "Minimize Args": self.optimizers[name][2]}

                    data = {
                            "Assignment": {"Task": type(self.task.__wrapped__).__name__,
                                           "Working Dir": os.getcwd(),
                                           "Start Time": self.dt_start,
                                           "Stop Time": datetime.now()},
                            "Counters": {"Function Evaluations": self.f_counter,
                                         "Hunts Started": self.hunt_counter,
                                         "GPR Hyperparameter Optimisations": self.hyop_counter,
                                         "Optimizers": {"Started": self.o_counter,
                                                        "Killed": len(self.hunt_victims),
                                                        "Converged": self.conv_counter}},
                            "Solution": {"x": result.x,
                                         "fx": result.fx,
                                         "stats": result.stats,
                                         "origin": result.origin,
                                         "exit cond.": reason},
                            "Settings": {"x0 Generator": type(self.x0_generator).__name__,
                                         "Convergence Checker": type(self.convergence_checker).__name__,
                                         "Optimizers Available": optimizers,
                                         "Max Parallel Optimizers": self.max_jobs}
                            }
                    yaml.dump(data, file, default_flow_style=False)
                if self.history_logging == 3:
                    self.log.save("glompo_optimizer_logs")
                elif self.history_logging == 2 and best_id > 0:
                    self.log.save("glompo_best_optimizer_log", best_id)

            # Delete temp files or files not cleaned due to crashes
            # TODO Possible todo

            self._optional_print("-----------------------------------\n"
                                 "GloMPO Optimization Routine... DONE\n"
                                 "-----------------------------------\n")

    def _check_signals(self, opt_id: int):
        pipe = self.optimizer_packs[opt_id].signal_pipe
        self.last_feedback[opt_id] = time()
        if pipe.poll():
            key, message = pipe.recv()
            self._optional_print(f"\tSignal {key} from {opt_id}.")
            if key == 0:
                self.log.put_metadata(opt_id, "Stop Time", datetime.now())
                self.log.put_metadata(opt_id, "End Condition", message)
                self.graveyard.add(opt_id)
                self.conv_counter += 1
            elif key == 1:
                # TODO Deal with 1 signals
                pass
            elif key == 9:
                self.log.put_message(opt_id, message)
        else:
            self._optional_print(f"\tNo signals from {opt_id}.")

    def _start_new_job(self, opt_id: int, optimizer: BaseOptimizer, call_kwargs: Dict[str, Any],
                       pipe: Connection, event: Event, gpr: GaussianProcessRegression):

        self._optional_print(f"Starting Optimizer: {opt_id}")

        task = self.task
        x0 = self.x0_generator.generate()
        bounds = np.array(self.bounds)

        process = mp.Process(target=optimizer.minimize,
                             args=(task, x0, bounds),
                             kwargs=call_kwargs,
                             daemon=True)

        self.optimizer_packs[opt_id] = ProcessPackage(process, pipe, event, gpr)
        self.optimizer_packs[opt_id].process.start()
        self.last_feedback[opt_id] = time()

        if self.visualisation:
            if opt_id not in self.scope.streams:
                self.scope.add_stream(opt_id)

    def _shutdown_job(self, opt_id):
        self.hunt_victims[opt_id] = time()

        self.optimizer_packs[opt_id].signal_pipe.send(1)
        self._optional_print(f"Termination signal sent to {opt_id}")

        self.log.put_metadata(opt_id, "Stop Time", datetime.now())
        self.log.put_metadata(opt_id, "End Condition", "GloMPO Termination")

        if self.visualisation:
            self.scope.update_kill(opt_id)

    def _start_hunt(self):
        self._optional_print(f"Starting hunt")
        self.hunt_counter += 1

        log = self.log
        gprs = {}
        for opt_id in self.optimizer_packs:
            if self.optimizer_packs[opt_id].process.is_alive():
                gprs[opt_id] = self.optimizer_packs[opt_id].gpr

        def wrapped_hunt(queue, **kwargs):
            result = self._hunt(**kwargs)
            queue.put(result)

        process = mp.Process(target=wrapped_hunt,
                             args=(self.hunting_queue,),
                             kwargs={"hunt_id": self.hunt_counter,
                                     "log": log,
                                     "gprs": gprs},
                             daemon=True)

        self.hunting_processes[self.hunt_counter] = process
        self.hunting_processes[self.hunt_counter].start()

    def _start_hyperparam_job(self, opt_id):
        self._optional_print(f"Starting hyperparameter optimization job for {opt_id}")

        self.hyop_counter += 1
        gpr = self.optimizer_packs[opt_id].gpr

        # TODO Smarter selection of optimization parameters
        kernel_kwargs = {"time_series": gpr.training_coords(),
                         "loss_series": gpr.training_values(),
                         "noise": True,
                         "bounds": None,
                         "x0": None,
                         "verbose": True}

        def wrapped_hyperparm_job(hyper_id, o_id, queue, **kwargs):
            a, b, g = self.optimizer_packs[o_id].gpr.kernel.optimize_hyperparameters(**kwargs)
            result = HyperparameterOptResult(hyper_id, o_id, a, b, g)
            queue.put(result)

        process = mp.Process(target=wrapped_hyperparm_job,
                             args=(self.hyop_counter, opt_id, self.hyperparm_queue),
                             kwargs=kernel_kwargs,
                             daemon=True)

        self.hyperparm_processes[self.hyop_counter] = process
        self.hyperparm_processes[self.hyop_counter].start()

        if self.visualisation:
            self.scope.update_opt_start(opt_id)

    def _explore_basin(self):
        pass

    def _setup_new_optimizer(self):
        self.o_counter += 1

        # TODO add intelligence to pick optimizer?
        selected, init_kwargs, call_kwargs = self.optimizers['default']

        self._optional_print(f"Selected Optimizer:\n"
                             f"\tOptimizer ID: {self.o_counter}\n"
                             f"\tType: {selected.__name__}")

        gpr = GaussianProcessRegression(kernel=ExpKernel(alpha=0.100,
                                                         beta=5.00),
                                        dims=1,
                                        sigma_noise=0,
                                        mean=0)

        parent_pipe, child_pipe = mp.Pipe()
        event = self.mp_manager.Event()
        event.set()

        optimizer = selected(**init_kwargs,
                             opt_id=self.o_counter,
                             signal_pipe=child_pipe,
                             results_queue=self.optimizer_queue,
                             pause_flag=event)

        self.log.add_optimizer(self.o_counter, type(optimizer).__name__, datetime.now())

        return OptimizerPackage(self.o_counter, optimizer, call_kwargs, parent_pipe, event, gpr)

    def _optional_print(self, message: str):
        if self.verbose:
            print(message)

    @staticmethod
    def _hunt(hunt_id: int, log: Logger, gprs: Dict[int, GaussianProcessRegression]) -> HuntingResult:
        victims = set()
        for gpr_id in gprs:
            for log_id in log.storage:
                if gpr_id != log_id:

                    # TODO Better selection of the 'when' to sample, currently: 1000 + furthest point so far
                    history = log.get_history(log_id, "fx_best")
                    i = len(history)
                    best = history[-1]
                    mean, std = gprs[gpr_id].sample_all(i+1000)

                    if best < mean - 2 * std:
                        victims.add(gpr_id)

        result = HuntingResult(hunt_id, victims)
        return result
