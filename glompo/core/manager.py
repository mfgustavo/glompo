

# Native Python imports
import shutil
import sys
import warnings
import multiprocessing as mp
import traceback
import os
from datetime import datetime
from functools import wraps
from time import time
from typing import *

# Package imports
from ..generators.basegenerator import BaseGenerator
from ..generators.random import RandomGenerator
from ..convergence.nkillsafterconv import KillsAfterConvergence
from ..convergence.basechecker import BaseChecker
from ..common.namedtuples import *
from ..common.customwarnings import *
from ..hunters import BaseHunter, ValBelowGPR, PseudoConverged, MinVictimTrainingPoints, GPRSuitable
from ..optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from .gpr import GaussianProcessRegression
from .expkernel import ExpKernel
from .logger import Logger
from .scope import GloMPOScope

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
                 gpr_training: Tuple[Union[int, None], int] = (None, 10),
                 history_logging: int = 0,
                 visualisation: bool = False,
                 visualisation_args: Optional[Dict[str, Any]] = None,
                 force_terminations_after: int = -1,
                 verbose: int = 0,
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
            Sequence of tuples of the form (min, max) limiting the range of each parameter. Do not use bounds to fix
            a parameter value as this will raise an error. Rather supply fixed parameter values through task_args or
            task_kwargs.

        working_dir: Optional[str] = None
            If provided, GloMPO wil redirect its outputs to the given directory.

        overwrite_existing: bool = False
            If True, GloMPO will overwrite existing files if any are found in the working_dir otherwise it will raise a
            FileExistsError if these results are detected.

        max_jobs: Optional[int] = None
            The maximum number of local optimizers run in parallel at one time. Defaults to one less than the number of
            CPUs available to the system with a minimum of 1.

        task_args: Optional[Tuple]] = None
            Optional arguments passed to task with every call.

        task_kwargs: Optional[Dict] = None
            Optional keyword arguments passed to task with every call.

        convergence_checker: Optional[BaseChecker] = None
            Criteria used for convergence. A collection of subclasses of BaseChecker are provided, tthese can be
            used in combinations of and (&) and or (|) to tailor various conditions.
                E.g.: convergence_criteria = MaxFuncCalls(20000) | KillsAfterConvergence(3, 1) & MaxSeconds(60*5)
                In this case GloMPO will run until 20 000 function evaluations OR until 3 optimizers have been killed
                after the first convergence provided it has at least run for five minutes.
            Default: KillsAfterConvergence(0, 1) i.e. GloMPO terminates as soon as any optimizer converges.


        x0_generator: Optional[BaseGenerator] = None
            An instance of a subclass of BaseGenerator which produces starting points for the optimizer. If not provided
            a random generator is used.

        killing_conditions: Optional[BaseHunter] = None
            Criteria used for killing optimizers. A collection of subclasses of BaseHunter are provided, these can be
            used in combinations of and (&) and or (|) to tailor various conditions.
                E.g.: killing_conditions = GPRSuitable() & ValBelowGPR() &
                (MinVictimTrainingPoints(30) & PseudoConverged(20))
                In this case GloMPO will only allow a hunt to terminate an optimizer if
                    1) the GPR describing it is a good descriptor of the data,
                    2) another optimizer has seen a value below its 95% confidence interval,
                    3) and the optimizer has either at least 30 training points or has not changed its best value in
                    20 iterations.
                Default: ValBelowGPR() & PseudoConverged(20, 0.01) & MinVictimTrainingPoints(10) & GPRSuitable(0.1)

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

        gpr_training: Tuple[Union[int, None], int] = (None, 10)
            Tuple of the form (max, step) which sets the maximum number of points in an optimizer GPR and step which
            controls the interval at which steps are taken. The later points are kept and earlier points are
            discarded from the regression.

        history_logging: int = 0
            Indicates the level of logging the user would like:
                0 - No log files are saved.
                1 - Only the manager log is saved.
                2 - The manager log and log file of the optimizer from which the final solution was extracted is saved.
                3 - The manager log and the log files of every started optimizer is saved.

        visualisation: bool = False
            If True then a dynamic plot is generated to demonstrate the performance of the optimizers. Further options
            (see visualisation_args) allow this plotting to be recorded and saved as a film.

        visualisation_args: Optional[Dict[str, Any]] = None
            Optional arguments to parameterize the dynamic plotting feature. See GloMPOScope.

        force_terminations_after: int = -1
            If a value larger than zero is provided then GloMPO is allowed to force terminate optimizers that have
            either not provided results in the provided number of seconds or optimizers which were sent a kill
            signal have not shut themselves down within the provided number of seconds.

            Use with caution: This runs the risk of corrupting the results queue but ensures resources are not wasted on
            hanging processes.

        verbose: int = 0
            An integer describing the amount of output generated by GloMPO:
                0 - Nothing is printed
                1 - Only some status messages are printed
                2 - Status messages are constantly printed. Useful for traceback during debugging.

        split_printstreams: bool = True
            If True, optimizer print messages will be intercepted and saved to separate files.
        """

        # CONSTANTS
        self._SIGNAL_DICT = {0: "Normal Termination (Args have reason)",
                             1: "I have made some nan's... oopsie... >.<"}
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("always", RuntimeWarning)
        warnings.simplefilter("always", NotImplementedWarning)

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
            except (FileNotFoundError, NotADirectoryError):
                try:
                    os.makedirs(working_dir)
                    os.chdir(working_dir)
                except TypeError:
                    warnings.warn(f"Cannot parse working_dir = {working_dir}. str or bytes expected. Using current "
                                  f"work directory.", UserWarning)

        # Save and wrap task
        if not callable(task):
            raise TypeError(f"{task} is not callable.")
        if not isinstance(task_args, list) and task_args is not None:
            raise TypeError(f"{task_args} cannot be parsed, list needed.")
        if not isinstance(task_kwargs, dict) and task_kwargs is not None:
            raise TypeError(f"{task_kwargs} cannot be parsed, dict needed.")

        if not task_args:
            task_args = ()
        if not task_kwargs:
            task_kwargs = {}
        self.task = self._task_args_wrapper(task, task_args, task_kwargs)

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
            elif len(optimizers[key]) == 3 and \
                    (isinstance(optimizers[key][1], dict) or optimizers[key][1] is None) and \
                    (isinstance(optimizers[key][2], dict) or optimizers[key][2] is None):
                init = {} if optimizers[key][1] is None else optimizers[key][1]
                call = {} if optimizers[key][2] is None else optimizers[key][2]
                optimizers[key] = (optimizers[key][0], init, call)
            else:
                raise ValueError(f"Cannot parse {optimizers[key]}.")
            if not issubclass(optimizers[key][0], BaseOptimizer):
                raise TypeError(f"{optimizers[key][0]} not an instance of BaseOptimizer.")
        self.optimizers = optimizers
        # TODO Implement
        if any([key in optimizers for key in ['noisy', 'late', 'early']]):
            warnings.warn("Optimizer keys other than 'default' are not currently supported and will be ignored.",
                          NotImplementedWarning)

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
            self.max_jobs = mp.cpu_count() - 1

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
            self.killing_conditions = ValBelowGPR() & \
                                      PseudoConverged(20, 0.01) & \
                                      MinVictimTrainingPoints(10) & \
                                      GPRSuitable(0.2)

        # Save GPR control
        if not isinstance(gpr_training, tuple):
            raise TypeError(f"Cannot parse gpr_training, Tuple[int, int] required.")
        else:
            self.gpr_max = int(gpr_training[0]) if gpr_training[0] is not None else np.inf
            self.gpr_step = int(gpr_training[1])

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
            self.scope = GloMPOScope(**visualisation_args) if visualisation_args else GloMPOScope()
        # TODO Implement
        if region_stability_check:
            warnings.warn("region_stbility_check not implemented. Ignoring.", NotImplementedWarning)
        # TODO Implement
        if report_statistics:
            warnings.warn("report_statistics not implemented. Ignoring.", NotImplementedWarning)

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
        """ Begins the optimization routine and returns the selected minimum in an instance of MinimizeResult. """

        # Variables needed outside loop
        best_id = 0
        result = Result(None, None, None, None)
        converged = False
        reason = ""
        caught_exception = None

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

                # Force kill any zombies
                if self.allow_forced_terminations:
                    for id_num in self.hunt_victims:
                        if time() - self.hunt_victims[id_num] > self._TOO_LONG and \
                                self.optimizer_packs[id_num].process.is_alive():
                            self.optimizer_packs[id_num].process.terminate()
                            self.optimizer_packs[id_num].process.join()
                            self.log.put_message(id_num, "Force terminated due to no feedback after kill signal "
                                                         "timeout.")
                            self.log.put_metadata(id_num, "Approximate Stop Time", datetime.now())
                            self.log.put_metadata(id_num, "End Condition", "Forced GloMPO Termination")
                            warnings.warn(f"Forced termination signal sent to optimizer {id_num}.", RuntimeWarning)

                # Check results_queue
                self._optional_print("Checking optimizer iteration results...", 2)
                i_count = 0
                while not self.optimizer_queue.empty() and i_count < 10:
                    res = self.optimizer_queue.get()
                    i_count += 1
                    self.last_feedback[res.opt_id] = time()
                    self.f_counter += res.n_icalls

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
                        if res.n_iter % self.gpr_step == 0:
                            trained = True
                            self._optional_print(f"\tResult from {res.opt_id} sent to GPR", 2)

                            gpr = self.optimizer_packs[res.opt_id].gpr
                            gpr.add_known(res.n_iter, fx)
                            while len(gpr.training_coords()) > self.gpr_max:
                                gpr.remove(gpr.training_coords()[0])
                            if self.visualisation:
                                x = np.transpose(self.scope.streams[res.opt_id]['train_pts'].get_data())
                                while len(x) > self.gpr_max:
                                    x = np.delete(x, 0, axis=0)
                                self.scope.streams[res.opt_id]['train_pts'].set_data(np.transpose(x))

                            cut = self.gpr_max * self.gpr_step if self.gpr_max != np.inf else 0
                            mean = np.mean(self.log.get_history(res.opt_id, "fx")[-cut:])
                            sigma = np.std(self.log.get_history(res.opt_id, "fx")[-cut:])
                            sigma = 1 if sigma == 0 else sigma

                            self.optimizer_packs[res.opt_id].gpr.rescale((mean, sigma))

                            # Start new hyperparameter optimization job
                            if res.opt_id not in self.hyperparm_processes and res.n_iter > 0:
                                if not self.optimizer_packs[res.opt_id].gpr.is_suitable():
                                    self._start_hyperparam_job(res.opt_id)

                            # Hunt optimizers
                            self._optional_print("Hunting...", 2)
                            for hunter_id in self.optimizer_packs:
                                for victim_id in self.optimizer_packs:
                                    ids = [hunter_id, victim_id]
                                    in_graveyard = any([opt_id in self.graveyard for opt_id in ids])
                                    in_update = any([opt_id in self.hyperparm_processes for opt_id in ids])
                                    has_points = all([len(self.log.get_history(opt_id, "fx")) > 0 for opt_id in ids])
                                    has_gpr = all([len(self.optimizer_packs[opt_id].gpr.training_coords()) > 0
                                                   for opt_id in ids])
                                    if not in_graveyard and not in_update and has_points and has_gpr:
                                        self.hunt_counter += 1
                                        kill = self.killing_conditions.is_kill_condition_met(self.log,
                                                                                             hunter_id,
                                                                                             self.optimizer_packs[
                                                                                                 hunter_id].gpr,
                                                                                             victim_id,
                                                                                             self.optimizer_packs[
                                                                                                 victim_id].gpr)
                                        if kill:
                                            self._optional_print(f"\tManager wants to kill {victim_id}", 1)
                                            if victim_id not in self.graveyard:
                                                self._shutdown_job(victim_id)
                            self._optional_print("Hunt check done.", 2)

                        if self.visualisation:
                            self.scope.update_optimizer(res.opt_id, (self.f_counter, fx))
                            if trained:
                                self.scope.update_scatter(res.opt_id, (self.f_counter, fx))

                                if self.scope.visualise_gpr:
                                    gpr = self.optimizer_packs[res.opt_id].gpr
                                    i_min = np.clip(res.n_iter - self.gpr_max * self.gpr_step, 1, None)
                                    i_max = res.n_iter + 100
                                    i_range = np.linspace(i_min, i_max, 200)
                                    mu, sigma = gpr.sample_all(i_range)

                                    f_min = self.log.get_history(res.opt_id, "f_call")[i_min]
                                    f_max = self.f_counter + self.max_jobs * 100
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
                                              f"minimization complete signal to the manager.", RuntimeWarning)
                                self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                                self.log.put_metadata(opt_id, "End Condition", "Normal termination "
                                                                               "(Reason unknown)")
                        else:
                            self.graveyard.add(opt_id)
                            self.log.put_message(opt_id, f"Terminated in error with code {-exitcode}")
                            warnings.warn(f"Optimizer {opt_id} terminated in error with code {-exitcode}",
                                          RuntimeWarning)
                            self.log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                            self.log.put_metadata(opt_id, "End Condition", f"Error termination (exitcode {-exitcode}).")
                    if self.optimizer_packs[opt_id].process.is_alive() and time() - self.last_feedback[opt_id] > \
                            self._TOO_LONG and self.allow_forced_terminations and opt_id not in self.hunt_victims:
                        warnings.warn(f"Optimizer {opt_id} seems to be hanging. Forcing termination.", RuntimeWarning)
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
                        self.hyperparm_processes[res.opt_id].join()
                        del self.hyperparm_processes[res.opt_id]

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
                for opt_id in self.optimizer_packs:
                    history = self.log.get_history(opt_id, "fx_best")
                    if len(history) > 0:
                        opt_best = history[-1]
                        if opt_best < best_fx:
                            best_fx = opt_best
                            best_id = opt_id

                            i = self.log.get_history(opt_id, "i_best")[-1]
                            best_x = self.log.get_history(opt_id, "x")[i-1]
                            best_origin = {"opt_id": opt_id,
                                           "type": self.log.get_metadata(opt_id, "Optimizer Type")}

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

        except KeyboardInterrupt:
            caught_exception = "User Interrupt"
            self._cleanup_crash("User Interrupt")

        except Exception as e:
            caught_exception = "".join(traceback.TracebackException.from_exception(e).format())
            warnings.warn(f"Optimization failed. Caught exception: {e}", RuntimeWarning)
            print(caught_exception)
            self._cleanup_crash("GloMPO Crash")

        finally:

            # Make movie
            if self.visualisation and self.scope.record_movie and not caught_exception:
                self.scope.generate_movie()

            # Save log
            if self.history_logging > 0:
                if caught_exception:
                    reason = f"Process Crash: {caught_exception}"
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

            return result

    def _cleanup_crash(self, opt_reason: str):
        for opt_id in self.optimizer_packs:
            self.graveyard.add(opt_id)
            self.log.put_metadata(opt_id, "Stop Time", datetime.now())
            self.log.put_metadata(opt_id, "End Condition", opt_reason)
            self.optimizer_packs[opt_id].process.join(1)
            if self.optimizer_packs[opt_id].process.is_alive():
                self.optimizer_packs[opt_id].process.terminate()
            if self.hyperparm_processes[opt_id].process.is_alive():
                self.hyperparm_processes[opt_id].process.terminate()

    def _check_signals(self, opt_id: int) -> bool:
        """ Checks for signals from optimizer opt_id and processes it.
            Returns a bool indicating whether a signal was found.
        """
        pipe = self.optimizer_packs[opt_id].signal_pipe
        found_signal = False
        if opt_id not in self.graveyard and pipe.poll():
            try:
                key, message = pipe.recv()
                self.last_feedback[opt_id] = time()
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
                       pipe: mp.connection.Connection, event: mp.Event, gpr: GaussianProcessRegression):
        """ Given an initialised optimizer and multiprocessing variables, this method packages them and starts a new
            process.
        """

        self._optional_print(f"Starting Optimizer: {opt_id}", 2)

        task = self.task
        x0 = self.x0_generator.generate()
        bounds = np.array(self.bounds)
        target = self._catch_interrupt_wrapper(optimizer.minimize)

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
        """ Sends a stop signal to optimizer opt_id and updates variables around its termination. """
        self.hunt_victims[opt_id] = time()
        self.graveyard.add(opt_id)

        self.optimizer_packs[opt_id].signal_pipe.send(1)
        self._optional_print(f"Termination signal sent to {opt_id}", 1)

        self.log.put_metadata(opt_id, "Stop Time", datetime.now())
        self.log.put_metadata(opt_id, "End Condition", "GloMPO Termination")

        if self.visualisation:
            self.scope.update_kill(opt_id)

    def _start_hyperparam_job(self, opt_id):
        """ Starts a new process to update the hyperparameters for optimizer opt_id. """
        self._optional_print(f"Starting hyperparameter optimization job for {opt_id}", 1)

        self.hyop_counter += 1

        def wrapped_hyperparm_job(o_id, queue: mp.Queue, gpr: GaussianProcessRegression):

            a = gpr.kernel.alpha
            b = gpr.kernel.beta
            g = gpr.sigma_noise

            while not gpr.is_tail_suitable() and gpr.sigma_noise < 0.2:
                gpr.sigma_noise *= 1.1
                g *= 1.1
            if not gpr.is_mean_suitable(0.3):
                sig_min = np.clip(0.5*g, 0.001, 0.25)
                res = gpr.kernel.optimize_hyperparameters(time_series=gpr.training_coords(),
                                                          loss_series=gpr.training_values(),
                                                          noise=True,
                                                          bounds=((0.5, 1.5), (10, 50), (sig_min, 0.3)),
                                                          x0=None,
                                                          verbose=self.verbose == 2)
                if res is not None:
                    a = res[0]
                    b = res[1]
                    g = res[2]

            result = HyperparameterOptResult(o_id, a, b, g)
            queue.put(result)

        process = mp.Process(target=self._catch_interrupt_wrapper(wrapped_hyperparm_job),
                             args=(opt_id, self.hyperparm_queue, self.optimizer_packs[opt_id].gpr),
                             daemon=True)

        self.hyperparm_processes[opt_id] = process
        self.hyperparm_processes[opt_id].start()

        if self.visualisation:
            self.scope.update_opt_start(opt_id)

    def _setup_new_optimizer(self) -> OptimizerPackage:
        """ Selects and initializes new optimizer and multiprocessing variables. Returns an OptimizerPackage which
            can be sent to _start_new_job to begin new process.
        """
        self.o_counter += 1

        selected, init_kwargs, call_kwargs = self.optimizers['default']

        self._optional_print(f"Selected Optimizer:\n"
                             f"\tOptimizer ID: {self.o_counter}\n"
                             f"\tType: {selected.__name__}", 1)

        gpr = GaussianProcessRegression(kernel=ExpKernel(alpha=1.6,
                                                         beta=100),
                                        dims=1,
                                        sigma_noise=0.4,
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
                1 - Important messages only, used for limited printing.
                2 - Unimportant messages when full printing is requested.
        """
        if level <= self.verbose:
            print(message)

    @staticmethod
    def _redirect(opt_id, func):
        """ Wrapper to redirect a process' output to a designated text file. """
        @wraps(func)
        def wrapper(*args, **kwargs):
            sys.stdout = open(f"glompo_optimizer_printstreams/{opt_id}_printstream.out", "w")
            sys.stderr = open(f"glompo_optimizer_printstreams/{opt_id}_printstream.err", "w")
            func(*args, **kwargs)
        return wrapper

    @staticmethod
    def _task_args_wrapper(func, args, kwargs):
        """ Wraps a task's args and kwargs into it so that it becomes only a function of one variable (the vector
            of parameter values).
        """
        @wraps(func)
        def wrapper(x):
            return func(x, *args, **kwargs)

        return wrapper

    @staticmethod
    def _catch_interrupt_wrapper(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print("Interrupt signal received. Child process stopping.")

        return wrapper
