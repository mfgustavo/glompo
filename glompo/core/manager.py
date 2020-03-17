

# Native Python imports
import shutil
import sys
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
from ..convergence.nkillsafterconv import KillsAfterConvergence
from ..convergence.basechecker import BaseChecker
from ..common.namedtuples import *
from ..hunters import BaseHunter, ValBelowGPR, PseudoConverged, MinVictimTrainingPoints, GPRSuitable
from ..scope.scope import GloMPOScope
from ..optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from .gpr import GaussianProcessRegression
from .expkernel import ExpKernel
from .logger import Logger

# Other Python packages
import numpy as np
import yaml


__all__ = ("GloMPOManager",)


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
                 working_dir: Optional[str] = None,
                 overwrite_existing: bool = False,
                 max_jobs: Optional[int] = None,
                 task_args: Optional[Tuple] = None,
                 task_kwargs: Optional[Dict] = None,
                 convergence_checker: Optional[BaseChecker] = None,
                 x0_generator: Optional[BaseGenerator] = None,
                 killing_conditions: Optional[BaseHunter] = None,
                 region_stability_check: bool = False,
                 report_statistics: bool = False,
                 enforce_elitism: bool = False,
                 history_logging: int = 0,
                 visualisation: bool = False,
                 visualisation_args: Optional[Dict[str, Any]] = None,
                 force_terminations_after: int = -1,
                 verbose: bool = False,
                 split_printstreams: bool = True):
        """
        Generates the environment for a globally managed parallel optimization job.

        Parameters
        ----------
        task: Callable[[Sequence[float]], float]
            Function to be minimized. Accepts a 1D sequence of parameter values and returns a single value.
            Note: Must be a standalone function which makes no modifications outside of itself.

        n_parms: int
            The number of parameters to be optimized.

        optimizers: Dict[str, Union[Type[BaseOptimizer], Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]]
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

        bounds: Sequence[Tuple[float, float]]
            Sequence of tuples of the form (min, max) limiting the range of each parameter.

        working_dir: Optional[str] = None
            If provided, GloMPO wil redirect its outputs to the given directory

        overwrite_existing: bool = False
            If True, GloMPO will overwrite existing files if any are found in the working_dir otherwise it will raise a
            FileExistsError if these results are detected.

        max_jobs: int = None
            The maximum number of local optimizers run in parallel at one time. Defaults to one less than the number of
            CPUs available to the system with a minimum of 1

        task_args: Optional[Tuple]] = None
            Optional arguments passed to task with every call.

        task_kwargs: Optional[Dict] = None
            Optional keyword arguments passed to task with every call.

        convergence_checker: Optional[BaseChecker] = None
            Criteria used for convergence. A collection of subclasses of BaseChecker are provided, these can be used in
            combination to tailor various exit conditions. Instances which are added together represent an 'or'
            combination and instances which are multiplied together represent an 'and' combination.
                E.g.: convergence_criteria = MaxFuncCalls(20000) + KillsAfterConvergence(3, 1) * MaxSeconds(60*5)
                In this case GloMPO will run until 20 000 function evaluations OR until 3 optimizers have been killed
                after the first convergence provided it has at least run for five minutes.


        x0_generator: Optional[BaseGenerator] = None
            An instance of a subclass of BaseGenerator which produces starting points for the optimizer. If not provided
            a random generator is used.

        killing_conditions: Optional[BaseHunter] = None
            Criteria used for killing optimizers. A collection of subclasses of BaseHunter are provided, these can be
            used in combination to tailor various conditions. Instances which are added together represent an 'or'
            combination and instances which are multiplied together represent an 'and' combination.
                E.g.: killing_conditions = GPRSuitable() * ValBelowGPR() *
                (MinVictimTrainingPoints(30) + PseudoConverged(20))
                In this case GloMPO will only allow a hunt to terminate an optimizer if
                    1) the GPR describing it is a good descriptor of the data,
                    2) another optimizer has seen a value below its 95% confidence interval,
                    3) and the optimizer has either at least 30 training points or has not changed its best value in
                    20 iterations.

        region_stability_check: bool = False
            If True, local optimizers are started around a candidate solution which has been selected as a final
            solution. This is used to measure its reproducibility. If the check fails, the mp_manager resumes looking
            for another solution.

        report_statistics: bool = False
            If True, the mp_manager reports the statistical significance of the suggested solution.

        enforce_elitism: bool = False
            Some optimizers return their best ever results while some only return the result of a particular
            iteration. For particularly exploratory optimizers this can lead to large scale jumps in their evaluation
            of the error or trajectories which tend upward. This, in turn, can lead to poor predictive ability of the
            GPRs. If enforce_elitism is True, feedback from optimizers is filtered to only accept results which
            improve upon the incumbent.

        history_logging: int = 0
            Indicates the level of logging the user would like:
                0 - No log files are saved.
                1 - Only the manager log is saved.
                2 - The manager log and log file of the optimizer from which the final solution was extracted is saved.
                3 - The manager log and the log files of every started optimizer is saved.

        visualisation: bool
            If True then a dynamic plot is generated to demonstrate the performance of the optimizers. Further options
            (see visualisation_args) allow this plotting to be recorded and saved as a film.

        visualisation_args: Optional[Dict[str, Any]]
            Optional arguments to parameterize the dynamic plotting feature. See ParallelOptimizationScope.

        force_terminations_after: int = -1
            If a value larger than zero is provided then GloMPO is allowed to force terminate optimizers that have
            either not provided results in the provided number of seconds or optimizers which were sent a kill
            signal have not shut themselves down within the provided number of seconds.

            Use with caution: This runs the risk of corrupting the results queue but ensures resources are not wasted on
            hanging processes.

        verbose: int = 0
            An integer describing the amount of output generated by GloMPO:
                0 - Nothing is printed
                1 - Only critcal messages are printed
                2 - Status messages are constantly printed. Useful for traceback during debugging.

        split_printstreams: bool = True
            If True, optimizer print messages will be intercepted and saved to separate files.
        """

        # CONSTANTS
        self._SIGNAL_DICT = {0: "Normal Termination (Args have reason)",
                             1: "I have made some nan's... oopsie... >.<"}
        warnings.simplefilter("always", UserWarning)

        # Setup printing
        if isinstance(verbose, int):
            self.verbose = np.clip(int(verbose), 0, 2)
        else:
            raise ValueError(f"Cannot parse verbose = {verbose}. Only 0, 1, 2 are allowed.")
        self._optional_print("Initializing Manager ... ", 1)

        # Setup working directory
        if working_dir:
            try:
                os.chdir(working_dir)
            except FileNotFoundError:
                os.makedirs(working_dir)

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
        if isinstance(n_parms, int):
            if n_parms > 0:
                self.n_parms = n_parms
            else:
                raise ValueError(f"Cannot parse n_parms = {n_parms}. Only positive integers are allowed.")
        else:
            raise ValueError(f"Cannot parse n_parms = {n_parms}. Only integers are allowed.")

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
        if max_jobs:
            if isinstance(max_jobs, int):
                if max_jobs > 0:
                    self.max_jobs = max_jobs
                else:
                    raise ValueError(f"Cannot parse max_jobs = {max_jobs}. Only positive integers are allowed.")
            else:
                raise TypeError(f"Cannot parse max_jobs = {max_jobs}. Only positive integers are allowed.")
        else:
            self.max_jobs = mp.cpu_count()

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
                raise TypeError(f"x0_generator not an instance of a subclass of BaseGenerator.")
        else:
            self.x0_generator = RandomGenerator(self.bounds)

        # Save killing conditions
        if killing_conditions:
            if isinstance(killing_conditions, BaseHunter):
                self.killing_conditions = killing_conditions
            else:
                raise TypeError(f"killing_conditions not an instance of a subclass of BaseHunter.")
        else:
            self.killing_conditions = ValBelowGPR() * \
                                      PseudoConverged(20, 0.01) * \
                                      MinVictimTrainingPoints(10) * \
                                      GPRSuitable()

        # Save max conditions and counters
        self.t_start = None
        self.dt_start = None
        self.o_counter = 0
        self.f_counter = 0
        self.conv_counter = 0  # Number of _converged optimizers
        self.hunt_counter = 0  # Number of hunting jobs started
        self.hyop_counter = 0  # Number of hyperparameter optimization jobs started
        self.hunt_victims = {}  # opt_ids of killed jobs and timestamps when the signal was sent

        # Save behavioural args
        self.allow_forced_terminations = force_terminations_after > 0
        self._TOO_LONG = force_terminations_after
        self.region_stability_check = bool(region_stability_check)
        self.report_statistics = bool(report_statistics)
        self.enforce_elitism = bool(enforce_elitism)
        self.history_logging = np.clip(int(history_logging), 0, 3)
        self.log = Logger()
        self.visualisation = visualisation
        self.split_printstreams = split_printstreams
        if visualisation:
            self.scope = GloMPOScope(**visualisation_args)

        # Setup multiprocessing variables
        self.optimizer_packs = {}  # Dict[opt_id (int): ProcessPackage (NamedTuple)]
        self.hyperparm_processes = {}
        self.graveyard = set()  # opt_ids of known non-active optimizers
        self.last_feedback = {}

        self.mp_manager = mp.Manager()
        self.optimizer_queue = self.mp_manager.Queue()
        self.hyperparm_queue = self.mp_manager.Queue()

        # Purge Old Results
        files = os.listdir(".")
        if overwrite_existing:
            if any([file in files for file in ["glompo_manager_log.yml", "glompo_optimizer_logs",
                                               "glompo_best_optimizer_log"]]):
                self._optional_print("Old results found", 1)
                shutil.rmtree("glompo_manager_log.yml", ignore_errors=True)
                shutil.rmtree("glompo_optimizer_logs", ignore_errors=True)
                shutil.rmtree("glompo_best_optimizer_log", ignore_errors=True)
                shutil.rmtree("glompo_optimizer_printstreams", ignore_errors=True)
                self._optional_print("\tDeleted old results.", 1)
        else:
            raise FileExistsError("Previous results found. Remove, move or rename them. Alternatively, select another "
                                  "working_dir or set overwrite_existing=True.")

        if split_printstreams:
            os.makedirs("glompo_optimizer_printstreams", exist_ok=True)

        self._optional_print("Initialization Done", 1)

    def start_manager(self) -> MinimizeResult:
        """ Begins the optimization routine. """

        # Variables needed outside loop
        best_id = 0
        result = Result(None, None, None, None)
        converged = False
        reason = ""
        caught_exception = False

        try:
            self._optional_print("------------------------------------\n"
                                 "Starting GloMPO Optimization Routine\n"
                                 "------------------------------------\n", 1)

            self.t_start = time()
            self.dt_start = datetime.now()

            for i in range(self.max_jobs):
                opt = self._setup_new_optimizer()
                self._start_new_job(*opt)

            while not converged:

                # Start new processes if possible
                self._optional_print("Checking for available optimizer slots...", 2)
                processes = [pack.process for pack in self.optimizer_packs.values()]
                count = sum([int(proc.is_alive()) for proc in processes])
                while count < self.max_jobs:
                    opt = self._setup_new_optimizer()
                    self._start_new_job(*opt)
                    count += 1
                self._optional_print("New optimizer check done.", 2)

                # Check signals
                self._optional_print("Checking optimizer signals...", 2)
                for opt_id in self.optimizer_packs:
                    self._check_signals(opt_id)
                self._optional_print(f"Signal check done.", 2)

                # Hunt optimizers
                self._optional_print("Hunting...", 2)
                for hunter_id in self.optimizer_packs:
                    for victim_id in self.optimizer_packs:
                        if all([opt_id not in self.graveyard for opt_id in [hunter_id, victim_id]]) and \
                           all([len(self.log.get_history(opt_id, "fx")) > 10 for opt_id in [hunter_id, victim_id]]):
                            self.hunt_counter += 1
                            kill = self.killing_conditions.is_kill_condition_met(self.log,
                                                                                 hunter_id,
                                                                                 self.optimizer_packs[hunter_id].gpr,
                                                                                 victim_id,
                                                                                 self.optimizer_packs[victim_id].gpr)
                            if kill:
                                self._optional_print(f"\tManager wants to kill {victim_id}", 1)
                                self._shutdown_job(victim_id)
                self._optional_print("Hunt check done.", 2)

                # Force kill any zombies
                if self.allow_forced_terminations:
                    for id_num in self.hunt_victims:
                        if time() - self.hunt_victims[id_num] > self._TOO_LONG and \
                                self.optimizer_packs[id_num].process.is_alive():
                            self.optimizer_packs[id_num].process.terminate()
                            self.log.put_message(id_num, "Force terminated due to no feedback after kill signal "
                                                         "timeout.")
                            self.log.put_metadata(id_num, "Approximate Stop Time", datetime.now())
                            self.log.put_metadata(id_num, "End Condition", "Forced GloMPO Termination")
                            warnings.warn(f"Forced termination signal sent to optimizer {id_num}.", UserWarning)

                # Check results_queue
                self._optional_print("Checking optimizer iteration results...", 2)
                i_count = 0
                while not self.optimizer_queue.empty() and i_count < 10:
                    res = self.optimizer_queue.get()
                    i_count += 1
                    self.last_feedback[res.opt_id] = time()
                    self.f_counter += res.n_icalls
                    if self.convergence_checker.check_convergence(self):
                        break
                    if not any([res.opt_id == victim for victim in self.hunt_victims]):

                        # Apply elitism
                        fx = res.fx
                        if self.enforce_elitism:
                            history = self.log.get_history(res.opt_id, "fx_best")
                            if len(history) > 0:
                                if history[-1] < fx:
                                    fx = history[-1]

                        self.x0_generator.update(res.x, fx)
                        self.log.put_iteration(res.opt_id, res.n_iter, self.f_counter, list(res.x), fx)
                        self._optional_print(f"\tResult from {res.opt_id} @ iter {res.n_iter} fx = {fx}", 2)

                        # Send results to GPRs
                        trained = False
                        if res.n_iter % 3 == 0:
                            trained = True
                            self._optional_print(f"\tResult from {res.opt_id} sent to GPR", 2)
                            self.optimizer_packs[res.opt_id].gpr.add_known(res.n_iter, fx)

                            # Start new hyperparameter optimization job
                            # TODO Restart hyperparam jobs
                            # if len(self.optimizer_packs[res.opt_id].gpr.training_values()) % 20 == 0:
                            #     self._start_hyperparam_job(res.opt_id)

                            mean = np.mean(self.log.get_history(res.opt_id, "fx"))
                            sigma = np.std(self.log.get_history(res.opt_id, "fx"))
                            sigma = 1 if sigma == 0 else sigma

                            self.optimizer_packs[res.opt_id].gpr.rescale((mean, sigma))

                        if self.visualisation:
                            self.scope.update_optimizer(res.opt_id, (self.f_counter, fx))
                            if trained:
                                self.scope.update_scatter(res.opt_id, (self.f_counter, fx))

                                if self.scope.visualise_gpr:
                                    i_range = np.linspace(0, res.n_iter, 200)
                                    mu, sigma = self.optimizer_packs[res.opt_id].gpr.sample_all(i_range)

                                    f_range = self.scope.streams[res.opt_id]['all_opt'].get_xdata()
                                    f_min = self.log.get_history(res.opt_id, "f_call")[0]
                                    f_max = np.max(f_range)
                                    f_range = np.linspace(f_min, f_max, 200, endpoint=True)

                                    self.scope.update_gpr(res.opt_id, f_range, mu, mu - 2*sigma, mu + 2*sigma)
                                else:
                                    mu, sigma = self.optimizer_packs[res.opt_id].gpr.estimate_mean()
                                    self.scope.update_mean(res.opt_id, mu, sigma)
                            if res.final:
                                self.scope.update_norm_terminate(res.opt_id)
                else:
                    self._optional_print("\tNo results found.", 2)
                self._optional_print("Iteration results check done.", 2)

                # Check processes' statuses
                for opt_id in self.optimizer_packs:
                    if opt_id not in self.graveyard and not self.optimizer_packs[opt_id].process.is_alive():
                        exitcode = self.optimizer_packs[opt_id].process.exitcode
                        if exitcode == 0:
                            if not self._check_signals(opt_id):
                                self.conv_counter += 1
                                self.graveyard.add(opt_id)
                                self.log.put_message(opt_id, "Terminated normally without sending a minimization "
                                                             "complete signal to the manager.")
                                warnings.warn(f"Optimizer {opt_id} terminated normally without sending a "
                                              f"minimization complete signal to the manager.", UserWarning)
                                self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                                self.log.put_metadata(opt_id, "End Condition", "Normal termination "
                                                                               "(Reason unknown)")
                        else:
                            self.graveyard.add(opt_id)
                            self.log.put_message(opt_id, f"Terminated in error with code {-exitcode}")
                            warnings.warn(f"Optimizer {opt_id} terminated in error with code {-exitcode}", UserWarning)
                            self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                            self.log.put_metadata(opt_id, "End Condition", f"Error termination (exitcode {-exitcode}).")
                    if self.optimizer_packs[opt_id].process.is_alive() and time() - self.last_feedback[opt_id] > \
                            self._TOO_LONG and self.allow_forced_terminations:
                        warnings.warn(f"Optimizer {opt_id} seems to be hanging. Forcing termination.", UserWarning)
                        self.graveyard.add(opt_id)
                        self.log.put_message(opt_id, "Force terminated due to no feedback timeout.")
                        self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                        self.log.put_metadata(opt_id, "End Condition", "Forced GloMPO Termination")
                        self.optimizer_packs[opt_id].process.terminate()

                # Check hyperparm queue
                self._optional_print("Checking for _converged hyperparameter optimizations...", 2)
                if not self.hyperparm_queue.empty():
                    while not self.hyperparm_queue.empty():
                        res = self.hyperparm_queue.get_nowait()
                        self.hyperparm_processes[res.hyper_id].join()
                        del self.hyperparm_processes[res.hyper_id]

                        self._optional_print(f"\tNew hyperparameters found for {res.opt_id}", 1)

                        self.optimizer_packs[res.opt_id].gpr.kernel.alpha = res.alpha
                        self.optimizer_packs[res.opt_id].gpr.kernel.beta = res.beta
                        self.optimizer_packs[res.opt_id].gpr.sigma_noise = res.sigma

                        if self.visualisation:
                            self.scope.update_opt_end(res.opt_id)
                else:
                    self._optional_print("\tNo results found.", 2)
                self._optional_print("New hyperparameter check done.", 2)

                # Check convergence
                converged = self.convergence_checker.check_convergence(self)
                if converged:
                    self._optional_print(f"Convergence Reached", 1)

                # Setup candidate solution
                # TODO Improve solution selection
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

            self._optional_print(f"Exiting manager loop.\n"
                                 f"Exit conditions met: \n{self.convergence_checker.is_converged_str()}\n", 1)
            reason = self.convergence_checker.is_converged_str().replace("\n", "")

            # Check answer
            # TODO Check answer

            # Join all processes
            for opt_id in self.optimizer_packs:
                if self.optimizer_packs[opt_id].process.is_alive():
                    self.optimizer_packs[opt_id].signal_pipe.send(1)
                    self.graveyard.add(opt_id)
                    self.log.put_metadata(opt_id, "Stop Time", datetime.now())
                    self.log.put_metadata(opt_id, "End Condition", f"GloMPO Convergence")
                    self.optimizer_packs[opt_id].process.join(self._TOO_LONG / 20)
            for hypopt in self.hyperparm_processes.values():
                hypopt.terminate()

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
                                         "Convergence Checker": str(self.convergence_checker).replace("\n", ""),
                                         "Hunt Conditions": str(self.killing_conditions).replace("\n", ""),
                                         "Optimizers Available": optimizers,
                                         "Max Parallel Optimizers": self.max_jobs}
                            }
                    yaml.dump(data, file, default_flow_style=False)
                if self.history_logging == 3:
                    self.log.save("glompo_optimizer_logs")
                elif self.history_logging == 2 and best_id > 0:
                    self.log.save("glompo_best_optimizer_log", best_id)

            self._optional_print("-----------------------------------\n"
                                 "GloMPO Optimization Routine... DONE\n"
                                 "-----------------------------------\n", 1)

    def _check_signals(self, opt_id: int):
        pipe = self.optimizer_packs[opt_id].signal_pipe
        self.last_feedback[opt_id] = time()
        found_signal = False
        if opt_id not in self.graveyard and pipe.poll():
            try:
                key, message = pipe.recv()
                self._optional_print(f"\tSignal {key} from {opt_id}.", 1)
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
                found_signal = True
            except EOFError:
                self._optional_print(f"\tOpt{opt_id} pipe closed. Opt{opt_id} should be in graveyard", 1)
        else:
            self._optional_print(f"\tNo signals from {opt_id}.", 2)
        return found_signal

    def _start_new_job(self, opt_id: int, optimizer: BaseOptimizer, call_kwargs: Dict[str, Any],
                       pipe: Connection, event: Event, gpr: GaussianProcessRegression):

        self._optional_print(f"Starting Optimizer: {opt_id}", 2)

        task = self.task
        x0 = self.x0_generator.generate()
        bounds = np.array(self.bounds)
        target = optimizer.minimize

        if self.split_printstreams:
            target = self._redirect(opt_id, optimizer.minimize)

        process = mp.Process(target=target,
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
        self.graveyard.add(opt_id)

        self.optimizer_packs[opt_id].signal_pipe.send(1)
        self._optional_print(f"Termination signal sent to {opt_id}", 1)

        self.log.put_metadata(opt_id, "Stop Time", datetime.now())
        self.log.put_metadata(opt_id, "End Condition", "GloMPO Termination")

        if self.visualisation:
            self.scope.update_kill(opt_id)

    def _start_hyperparam_job(self, opt_id):
        self._optional_print(f"Starting hyperparameter optimization job for {opt_id}", 1)

        self.hyop_counter += 1
        gpr = self.optimizer_packs[opt_id].gpr

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

        selected, init_kwargs, call_kwargs = self.optimizers['default']

        self._optional_print(f"Selected Optimizer:\n"
                             f"\tOptimizer ID: {self.o_counter}\n"
                             f"\tType: {selected.__name__}", 1)

        gpr = GaussianProcessRegression(kernel=ExpKernel(alpha=1.6,
                                                         beta=100),
                                        dims=1,
                                        sigma_noise=0.5,
                                        mean=None)

        parent_pipe, child_pipe = mp.Pipe()
        event = self.mp_manager.Event()
        event.set()

        optimizer = selected(**init_kwargs,
                             opt_id=self.o_counter,
                             signal_pipe=child_pipe,
                             results_queue=self.optimizer_queue,
                             pause_flag=event)

        self.log.add_optimizer(self.o_counter, type(optimizer).__name__, datetime.now())

        if call_kwargs:
            return OptimizerPackage(self.o_counter, optimizer, call_kwargs, parent_pipe, event, gpr)
        else:
            return OptimizerPackage(self.o_counter, optimizer, {}, parent_pipe, event, gpr)

    def _optional_print(self, message: str, level: int):
        """ Controls printing of messages according to the user's choice using the verbose setting.

            Parameters
            ----------
            message : str
                Message to be printed
            level : int
                0 - Critical messages which are always printed, overwriting user choice.
                1 - Important messages only, used for limited printing
                2 - Unimportant messages when full printing is requested
        """
        if level <= self.verbose:
            print(message)

    @staticmethod
    def _redirect(opt_id, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sys.stdout = open(f"glompo_optimizer_printstreams/{opt_id}_printstream.out", "w")
            sys.stderr = open(f"glompo_optimizer_printstreams/{opt_id}_printstream.err", "w")
            func(*args, **kwargs)
        return wrapper
