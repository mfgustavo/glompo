
""" Contains GloMPOManager class which is the main user interface for GloMPO. """


# Native Python imports
import logging
import shutil
import warnings
import multiprocessing as mp
import traceback
import os
import getpass
import socket
import sys
from datetime import datetime
from time import time
from typing import *

# Other Python packages
import numpy as np
import yaml

# Package imports
from ..generators import BaseGenerator, RandomGenerator
from ..convergence import BaseChecker, KillsAfterConvergence
from ..common.namedtuples import *
from ..common.customwarnings import *
from ..common.helpers import *
from ..common.wrappers import process_print_redirect, task_args_wrapper, catch_user_interrupt
from ..hunters import BaseHunter, PseudoConverged, TimeAnnealing, ValueAnnealing, ParameterDistance
from ..optimizers.baseoptimizer import BaseOptimizer
from ..opt_selectors.baseselector import BaseSelector
from .optimizerlogger import OptimizerLogger
from .regression import DataRegressor
from ._backends import CustomThread, ThreadPrintRedirect


__all__ = ("GloMPOManager",)


class GloMPOManager:
    """ Attempts to minimize a given function using numerous optimizers in parallel, based on their performance and
        decision criteria, will stop and intelligently restart others.
    """

    def __init__(self,
                 task: Callable[[Sequence[float]], float],
                 n_parms: int,
                 optimizer_selector: Optional[BaseSelector],
                 bounds: Sequence[Tuple[float, float]],
                 working_dir: Optional[str] = None,
                 overwrite_existing: bool = False,
                 max_jobs: Optional[int] = None,
                 task_args: Optional[Tuple] = None,
                 task_kwargs: Optional[Dict] = None,
                 backend: str = 'processes',
                 convergence_checker: Optional[BaseChecker] = None,
                 x0_generator: Optional[BaseGenerator] = None,
                 killing_conditions: Optional[BaseHunter] = None,
                 hunt_frequency: int = 100,
                 region_stability_check: bool = False,
                 report_statistics: bool = False,
                 enforce_elitism: bool = False,
                 summary_files: int = 0,
                 visualisation: bool = False,
                 visualisation_args: Optional[Dict[str, Any]] = None,
                 force_terminations_after: int = -1,
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

        optimizer_selector: Optional[BaseSelector]
            Selection criteria for new optimizers, must be an instance of a BaseSelector subclass. BaseSelector
            subclasses are initialised by default with a set of BaseOptimizer subclasses the user would like to make
            available to the optimization. See BaseSelector and BaseOptimizer documentation for more details.

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

        backend: str = 'processes'
            Indicates the form of parallelism used by the optimizers. 'processes' will bundle each optimizer into a
            multiprocessing.Process, while 'threads' will send the task to a threading.Thread.

            The appropriateness of each depends on the task itself. Using multiprocessing may provide computational
            advantages but becomes resource expensive as the task is duplicated between processes, there may also be
            I/O collisions if the task relies on external files during its calculation.

            If threads are used, make sure the task is thread-safe! Also note that forced terminations are not
            possible in this case and hanging optimizers will not be killed. The 'force_terminations_after' parameter
            is ignored.

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
                E.g.: killing_conditions = (PseudoConverged(100, 0.01) & TimeAnnealing(2) & ValueAnnealing()) |
                                           ParameterDistance(0.1)
                In this case GloMPO will only allow a hunt to terminate an optimizer if
                    1) an optimizer's best value has not improved by more than 1% in 100 function calls,
                    2) and it fails an annealing type test based on how many iterations it has run,
                    3) and if fails an annealing type test based on how far the victim's value is from the best
                    optimizer's best value,
                    4) or the two optimizers are iterating very close to one another in parameter space
                Default: (PseudoConverged(100, 0.01) & TimeAnnealing(2) & ValueAnnealing()) | ParameterDistance(0.1)
            Note, for performance and to allow conditionality between hunters conditions are evaluated 'lazily' i.e.
            x or y will return if x is True without evaluating y. x and y will return False if x is False without
            evaluating y.

        hunt_frequency: int = 100
            The number of function calls between successive attempts to evaluate optimizer performance and determine
            if they should be terminated.

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
            Bayesian regression. If enforce_elitism is True, feedback from optimizers is filtered to only accept
            results which improve upon the incumbent.

        summary_files: int = 0
            Indicates the level of saving the user would like:
                0 - No opt_log files are saved.
                1 - Only the manager summary file is saved.
                2 - The manager summary file and opt_log file of the optimizer from which the final solution was
                    extracted are saved.
                3 - The manager summary file and the opt_log files of every started optimizer are saved.

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

        split_printstreams: bool = True
            If True, optimizer print messages will be intercepted and saved to separate files.
        """

        # Filter Warnings
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("always", RuntimeWarning)
        warnings.simplefilter("always", NotImplementedWarning)

        # Setup logging
        self.logger = logging.getLogger('glompo.manager')
        self.logger.info("Initializing Manager ... ")

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
        self.task = task_args_wrapper(task, task_args, task_kwargs)
        self.logger.debug("Task wrapped successfully")

        # Save n_parms
        if isinstance(n_parms, int):
            if n_parms > 0:
                self.n_parms = n_parms
            else:
                raise ValueError(f"Cannot parse n_parms = {n_parms}. Only positive integers are allowed.")
        else:
            raise ValueError(f"Cannot parse n_parms = {n_parms}. Only integers are allowed.")

        # Save optimizer selection criteria
        if isinstance(optimizer_selector, BaseSelector):
            self.selector = optimizer_selector
        else:
            raise TypeError(f"optimizer_selector not an instance of a subclass of BaseSelector.")

        # Save bounds
        self.bounds: List[Bound] = []
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
            self.logger.info("max_jobs set to CPU count.")

        # Save convergence criteria
        if convergence_checker:
            if isinstance(convergence_checker, BaseChecker):
                self.convergence_checker = convergence_checker
            else:
                raise TypeError(f"convergence_checker not an instance of a subclass of BaseChecker.")
        else:
            self.convergence_checker = KillsAfterConvergence()
            self.logger.info("Convergence set to default: KillsAfterConvergence(0, 1)")

        # Save x0 generator
        if x0_generator:
            if isinstance(x0_generator, BaseGenerator):
                self.x0_generator = x0_generator
            else:
                raise TypeError(f"x0_generator not an instance of a subclass of BaseGenerator.")
        else:
            self.x0_generator = RandomGenerator(self.bounds)
            self.logger.info("x0 generator set to default: RandomGenerator()")

        # Save max conditions and counters
        self.t_start = None
        self.dt_start = None
        self.converged = False
        self.o_counter = 0
        self.f_counter = 0
        self.last_hunt = 0
        self.conv_counter = 0  # Number of converged optimizers
        self.hunt_counter = 0  # Number of hunting jobs started
        self.hunt_victims: Dict[int, float] = {}  # opt_ids of killed jobs and timestamps when the signal was sent

        # Save behavioural args
        self.allow_forced_terminations = force_terminations_after > 0
        self._TOO_LONG = force_terminations_after
        self.region_stability_check = bool(region_stability_check)
        self.report_statistics = bool(report_statistics)
        self.enforce_elitism = bool(enforce_elitism)
        self.summary_files = np.clip(int(summary_files), 0, 3)
        if self.summary_files != summary_files:
            self.logger.warning(f"summary_files argument given as {summary_files} clipped to {self.summary_files}")
        if summary_files:
            yaml.add_representer(LiteralWrapper, literal_presenter)
        self.split_printstreams = split_printstreams
        self.visualisation = visualisation
        self.hunt_frequency = hunt_frequency

        # Initialise support classes
        self.opt_log = OptimizerLogger()
        self.regressor = DataRegressor()
        if visualisation:
            from .scope import GloMPOScope  # Only imported if needed to avoid matplotlib compatibility issues
            self.scope = GloMPOScope(**visualisation_args) if visualisation_args else GloMPOScope()
        if region_stability_check:
            warnings.warn("region_stability_check not implemented. Ignoring.", NotImplementedWarning)
            self.logger.warning(f"region_stability_check not yet implemented")
        if report_statistics:
            warnings.warn("report_statistics not implemented. Ignoring.", NotImplementedWarning)
            self.logger.warning(f"report_statistics not yet implemented")

        # Save killing conditions
        if killing_conditions:
            if isinstance(killing_conditions, BaseHunter):
                self.killing_conditions = killing_conditions
            else:
                raise TypeError(f"killing_conditions not an instance of a subclass of BaseHunter.")
        else:
            self.killing_conditions = PseudoConverged(100, 0.01) & TimeAnnealing(2) & ValueAnnealing() | \
                                      ParameterDistance(self.bounds, 0.05)
            self.logger.info("Hunting conditions set to default: PseudoConverged(100, 0.01) & TimeAnnealing(2) & "
                             "ValueAnnealing() | ParameterDistance(bounds, 0.05)")

        # Setup backend
        if any([backend == valid_opt for valid_opt in ('processes', 'threads')]):
            self.proc_backend = backend == 'processes'
        else:
            self.backend = 'processes'
            self.logger.warning(f"Unable to parse backend '{backend}'. 'processes' or 'threads' expected."
                                f"Defaulting to 'processes'.")
            warnings.warn(f"Unable to parse backend '{backend}'. 'processes' or 'threads' expected."
                          f"Defaulting to 'processes'.")
        self.optimizer_packs: Dict[int, ProcessPackage] = {}
        self.graveyard: Set[int] = set()
        self.last_feedback: Dict[int, float] = {}

        self.mp_manager = mp.Manager()
        self.optimizer_queue = self.mp_manager.Queue()

        if self.split_printstreams and not self.proc_backend:
            sys.stdout = ThreadPrintRedirect(sys.stdout)
            sys.stderr = ThreadPrintRedirect(sys.stderr)

        if self.allow_forced_terminations and not self.proc_backend:
            warnings.warn("Cannot use force terminations with threading.", UserWarning)
            self.logger.warning("Cannot use force terminations with threading.")

        # Purge Old Results
        files = os.listdir(".")
        if overwrite_existing:
            if any([file in files for file in ["glompo_manager_log.yml", "glompo_optimizer_logs",
                                               "glompo_best_optimizer_log"]]):
                self.logger.debug("Old results found")
                shutil.rmtree("glompo_manager_log.yml", ignore_errors=True)
                shutil.rmtree("glompo_optimizer_logs", ignore_errors=True)
                shutil.rmtree("glompo_best_optimizer_log", ignore_errors=True)
                shutil.rmtree("glompo_optimizer_printstreams", ignore_errors=True)
                self.logger.warning("Deleted old results.")
        else:
            raise FileExistsError("Previous results found. Remove, move or rename them. Alternatively, select another "
                                  "working_dir or set overwrite_existing=True.")

        if split_printstreams:
            os.makedirs("glompo_optimizer_printstreams", exist_ok=True)

        self.logger.info("Initialization Done")

    def start_manager(self) -> Result:
        """ Begins the optimization routine and returns the selected minimum in an instance of MinimizeResult. """

        result = Result(None, None, None, None)
        reason = ""
        caught_exception = None
        best_id = -1

        try:
            self.logger.info("Starting GloMPO Optimization Routine")

            self.t_start = time()
            self.dt_start = datetime.now()

            while not self.converged:

                self.logger.debug("Checking for available optimizer slots")
                self._fill_optimizer_slots()
                self.logger.debug("New optimizer check done")

                self.logger.debug("Checking optimizer signals")
                for opt_id in self.optimizer_packs:
                    self._check_signals(opt_id)
                self.logger.debug(f"Signal check done.")

                self.logger.debug("Checking optimizer iteration results")
                self._process_results()
                self.logger.debug("Iteration results check done.")

                result = self._find_best_result()
                if result.origin and 'opt_id' in result.origin:
                    best_id = result.origin['opt_id']

                if best_id > 0:
                    self._start_hunt(best_id)

                if self.visualisation:
                    for opt_id in self.optimizer_packs:
                        if opt_id not in self.graveyard:
                            cache = self.regressor.get_mcmc_results(opt_id, 'asymptote')
                            if cache:
                                yn = self.opt_log.get_history(opt_id, "fx")[-1]
                                median, lower, upper = tuple(yn*i for i in cache)
                                self.scope.update_mean(opt_id, median, lower, upper)

                self.logger.debug("Checking for hanging processes")
                self._inspect_children()

                self.converged = self.convergence_checker(self)
                if self.converged:
                    self.logger.info(f"Convergence Reached")

            self.logger.info("Exiting manager loop")
            self.logger.info(f"Exit conditions met: \n"
                             f"{nested_string_formatting(self.convergence_checker.str_with_result())}")
            reason = self.convergence_checker.str_with_result()

            self.logger.debug("Cleaning up multiprocessing")
            self._stop_all_children()

            return result

        except KeyboardInterrupt:
            caught_exception = "User Interrupt"
            self.logger.error("Caught User Interrupt, closing GloMPO gracefully.")
            self._cleanup_crash("User Interrupt")

        except Exception as e:
            caught_exception = "".join(traceback.TracebackException.from_exception(e).format())
            self.logger.critical("Critical error encountered. Attempting to close GloMPO gracefully", exc_info=e)
            warnings.warn(f"Optimization failed. Caught exception: {e}", RuntimeWarning)
            self._cleanup_crash("GloMPO Crash")

        finally:

            self.logger.info("Cleaning up and closing GloMPO")

            if self.visualisation and self.scope.record_movie and not caught_exception:
                self.logger.debug("Generating movie")
                self.scope.generate_movie()

            self.logger.debug("Saving summary file results")
            self._save_log(best_id, result, reason, caught_exception)

            if not self.proc_backend and self.split_printstreams:
                sys.stdout.close()
                sys.stderr.close()

            self.logger.info("GloMPO Optimization Routine Done")

            return result

    def _fill_optimizer_slots(self):
        """ Starts new optimizers if there are slots available. """
        processes = [pack.process for pack in self.optimizer_packs.values()]
        count = sum([int(proc.is_alive()) for proc in processes])
        while count < self.max_jobs:
            opt = self._setup_new_optimizer()
            self._start_new_job(*opt)
            count += 1

    def _start_new_job(self, opt_id: int, optimizer: BaseOptimizer, call_kwargs: Dict[str, Any],
                       pipe: mp.connection.Connection, event: mp.Event):
        """ Given an initialised optimizer and multiprocessing variables, this method packages them and starts a new
            process.
        """

        self.logger.info(f"Starting Optimizer: {opt_id}")

        task = self.task
        x0 = self.x0_generator.generate()
        bounds = np.array(self.bounds)
        target = optimizer._minimize

        if self.split_printstreams and self.proc_backend:
            target = process_print_redirect(opt_id, optimizer._minimize)

        kwargs = {'target': target,
                  'args': (task, x0, bounds),
                  'kwargs': call_kwargs,
                  'name': f"Opt{opt_id}",
                  'daemon': True}
        if self.proc_backend:
            process = mp.Process(**kwargs)
        else:
            process = CustomThread(redirect_print=self.split_printstreams, **kwargs)

        self.optimizer_packs[opt_id] = ProcessPackage(process, pipe, event)
        self.optimizer_packs[opt_id].process.start()
        self.last_feedback[opt_id] = time()

        if self.visualisation:
            if opt_id not in self.scope.streams:
                self.scope.add_stream(opt_id, type(optimizer).__name__)

    def _setup_new_optimizer(self) -> OptimizerPackage:
        """ Selects and initializes new optimizer and multiprocessing variables. Returns an OptimizerPackage which
            can be sent to _start_new_job to begin new process.
        """
        self.o_counter += 1

        selected, init_kwargs, call_kwargs = self.selector.select_optimizer(self, self.opt_log)

        self.logger.info(f"Setting up optimizer {self.o_counter} of type {selected.__name__}")

        parent_pipe, child_pipe = mp.Pipe()
        event = self.mp_manager.Event()
        event.set()

        optimizer = selected(opt_id=self.o_counter,
                             signal_pipe=child_pipe,
                             results_queue=self.optimizer_queue,
                             pause_flag=event,
                             **init_kwargs)

        self.opt_log.add_optimizer(self.o_counter, type(optimizer).__name__, datetime.now())

        if call_kwargs:
            return OptimizerPackage(self.o_counter, optimizer, call_kwargs, parent_pipe, event)
        return OptimizerPackage(self.o_counter, optimizer, {}, parent_pipe, event)

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
                self.logger.info(f"Signal {key} from {opt_id}.")
                if key == 0:
                    self.opt_log.put_metadata(opt_id, "Stop Time", datetime.now())
                    self.opt_log.put_metadata(opt_id, "End Condition", message)
                    self.graveyard.add(opt_id)
                    self.conv_counter += 1
                elif key == 1:
                    # TODO Deal with 1 signals
                    pass
                elif key == 9:
                    self.opt_log.put_message(opt_id, message)
                    self.logger.info(f"Message received: {message}")
                found_signal = True
            except EOFError:
                self.logger.error(f"Opt{opt_id} pipe closed. Opt{opt_id} should be in graveyard")
        else:
            self.logger.debug(f"No signals from {opt_id}.")
        return found_signal

    def _inspect_children(self):
        """ Loops through all children processes and checks their status. Tidies up and gracefully deal with any
            strange behaviour such as crashes or non-responsive behaviour.
        """

        for opt_id in self.optimizer_packs:

            # Find dead optimzer processes that did not properly signal their termination.
            if opt_id not in self.graveyard and not self.optimizer_packs[opt_id].process.is_alive():
                exitcode = self.optimizer_packs[opt_id].process.exitcode
                if exitcode == 0:
                    if not self._check_signals(opt_id):
                        self.conv_counter += 1
                        self.graveyard.add(opt_id)
                        self.opt_log.put_message(opt_id, "Terminated normally without sending a minimization "
                                                         "complete signal to the manager.")
                        warnings.warn(f"Optimizer {opt_id} terminated normally without sending a "
                                      f"minimization complete signal to the manager.", RuntimeWarning)
                        self.logger.warning(f"Optimizer {opt_id} terminated normally without sending a "
                                            f"minimization complete signal to the manager.")
                        self.opt_log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                        self.opt_log.put_metadata(opt_id, "End Condition", "Normal termination (Reason unknown)")
                else:
                    self.graveyard.add(opt_id)
                    self.opt_log.put_message(opt_id, f"Terminated in error with code {-exitcode}")
                    warnings.warn(f"Optimizer {opt_id} terminated in error with code {-exitcode}",
                                  RuntimeWarning)
                    self.logger.error(f"Optimizer {opt_id} terminated in error with code {-exitcode}")
                    self.opt_log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                    self.opt_log.put_metadata(opt_id, "End Condition", f"Error termination (exitcode {-exitcode}).")

            # Find hanging processes
            if self.optimizer_packs[opt_id].process.is_alive() and \
                    time() - self.last_feedback[opt_id] > self._TOO_LONG and \
                    self.allow_forced_terminations and \
                    opt_id not in self.hunt_victims and \
                    self.proc_backend:
                warnings.warn(f"Optimizer {opt_id} seems to be hanging. Forcing termination.", RuntimeWarning)
                self.logger.error(f"Optimizer {opt_id} seems to be hanging. Forcing termination.")
                self.graveyard.add(opt_id)
                self.opt_log.put_message(opt_id, "Force terminated due to no feedback timeout.")
                self.opt_log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                self.opt_log.put_metadata(opt_id, "End Condition", "Forced GloMPO Termination")
                self.optimizer_packs[opt_id].process.terminate()

            # Force kill zombies
            if opt_id in self.hunt_victims and \
               self.allow_forced_terminations and \
               self.optimizer_packs[opt_id].process.is_alive() and \
               time() - self.hunt_victims[opt_id] > self._TOO_LONG and \
               self.proc_backend:
                self.optimizer_packs[opt_id].process.terminate()
                self.optimizer_packs[opt_id].process.join(3)
                self.opt_log.put_message(opt_id, "Force terminated due to no feedback after kill signal "
                                                 "timeout.")
                self.opt_log.put_metadata(opt_id, "Approximate Stop Time", datetime.now())
                self.opt_log.put_metadata(opt_id, "End Condition", "Forced GloMPO Termination")
                warnings.warn(f"Forced termination signal sent to optimizer {opt_id}.", RuntimeWarning)
                self.logger.error(f"Forced termination signal sent to optimizer {opt_id}.")

    def _process_results(self):
        """ Retrieve results from the queue and process them into the opt_log. """
        i_count = 0
        while not self.optimizer_queue.empty() and i_count < 10:
            res = self.optimizer_queue.get_nowait()
            i_count += 1
            self.last_feedback[res.opt_id] = time()
            self.f_counter += res.i_fcalls

            history = self.opt_log.get_history(res.opt_id, "f_call_opt")
            if len(history) > 0:
                opt_fcalls = history[-1] + res.i_fcalls
            else:
                opt_fcalls = res.i_fcalls

            if res.opt_id not in self.hunt_victims:

                # Apply elitism
                fx = res.fx
                if self.enforce_elitism:
                    history = self.opt_log.get_history(res.opt_id, "fx_best")
                    if len(history) > 0 and history[-1] < fx:
                        fx = history[-1]

                self.x0_generator.update(res.x, fx)
                self.opt_log.put_iteration(res.opt_id, res.n_iter, self.f_counter, opt_fcalls, list(res.x), fx)
                self.logger.debug(f"Result from {res.opt_id} @ iter {res.n_iter} fx = {fx}")

                if self.visualisation:
                    self.scope.update_optimizer(res.opt_id, (self.f_counter, fx))
                    if res.final:
                        self.scope.update_norm_terminate(res.opt_id)
            else:
                self.logger.debug("No results found.")

    def _start_hunt(self, hunter_id: int):
        """ Creates a new hunt with the provided hunter_id as the 'best' optimizer looking to terminate
            the other active optimizers according to the provided killing_conditions.
        """

        if self.f_counter - self.last_hunt > self.hunt_frequency:
            self.hunt_counter += 1
            self.last_hunt = self.f_counter

            if self.visualisation:
                self.scope.update_hunt_start(hunter_id)

            self.logger.debug("Starting hunt")
            self.logger.debug("XXX", self.optimizer_packs, self.opt_log)
            for victim_id in self.optimizer_packs:
                in_graveyard = victim_id in self.graveyard
                has_points = len(self.opt_log.get_history(victim_id, "fx")) > 0
                if not in_graveyard and has_points and victim_id != hunter_id:
                    self.logger.debug(f"Optimizer {hunter_id} -> Optimizer {victim_id} hunt started.")
                    kill = self.killing_conditions(self.opt_log, self.regressor, hunter_id, victim_id)

                    if kill:
                        reason = nested_string_formatting(self.killing_conditions.str_with_result())
                        self.logger.info(f"Optimizer {hunter_id} wants to kill Optimizer {victim_id}:\n"
                                         f"Reason:\n"
                                         f"{reason}")

                        if victim_id not in self.graveyard:
                            self._shutdown_job(victim_id, hunter_id, reason)

                        if self.visualisation:
                            self.scope.update_hunt_end(hunter_id)
            self.logger.debug("Hunting complete")

    def _shutdown_job(self, opt_id: int, hunter_id: int, reason: str):
        """ Sends a stop signal to optimizer opt_id and updates variables around its termination. """
        self.hunt_victims[opt_id] = time()
        self.graveyard.add(opt_id)

        self.optimizer_packs[opt_id].signal_pipe.send(1)
        self.logger.debug(f"Termination signal sent to {opt_id}")

        self.opt_log.put_metadata(opt_id, "Stop Time", datetime.now())
        self.opt_log.put_metadata(opt_id, "End Condition", LiteralWrapper(f"GloMPO Termination\n"
                                                                          f"Hunter: {hunter_id}\n"
                                                                          f"Reason: "
                                                                          f"{nested_string_formatting(reason)}"))

        if self.visualisation:
            self.scope.update_kill(opt_id)

    def _find_best_result(self) -> Result:
        # TODO Better answer selection esp in context of answer stability?
        best_fx = np.inf
        best_x = []
        best_stats = None
        best_origin = None

        for opt_id in self.optimizer_packs:
            history = self.opt_log.get_history(opt_id, "fx_best")
            if len(history) > 0:
                opt_best = history[-1]
                if opt_best < best_fx:
                    best_fx = opt_best
                    i = self.opt_log.get_history(opt_id, "i_best")[-1]
                    best_x = self.opt_log.get_history(opt_id, "x")[i - 1]
                    best_origin = {"opt_id": opt_id,
                                   "type": self.opt_log.get_metadata(opt_id, "Optimizer Type")}
                    self.logger.debug("Updated best result")

        return Result(best_x, best_fx, best_stats, best_origin)

    def _stop_all_children(self):
        for opt_id in self.optimizer_packs:
            if self.optimizer_packs[opt_id].process.is_alive():
                self.optimizer_packs[opt_id].signal_pipe.send(1)
                self.graveyard.add(opt_id)
                self.opt_log.put_metadata(opt_id, "Stop Time", datetime.now())
                self.opt_log.put_metadata(opt_id, "End Condition", f"GloMPO Convergence")
                self.optimizer_packs[opt_id].process.join(self._TOO_LONG / 20)
                self.logger.debug(f"Termination signal sent to optimizer {opt_id}")

    def _save_log(self, best_id: int, result: Result, reason: str, caught_exception: bool):
        if self.summary_files > 0:
            if caught_exception:
                reason = f"Process Crash: {caught_exception}"
            with open("glompo_manager_log.yml", "w") as file:

                data = {"Assignment": {"Task": type(self.task.__wrapped__).__name__,
                                       "Working Dir": os.getcwd(),
                                       "Username": getpass.getuser(),
                                       "Hostname": socket.gethostname(),
                                       "Start Time": self.dt_start,
                                       "Stop Time": datetime.now()},
                        "Settings": {"x0 Generator": type(self.x0_generator).__name__,
                                     "Convergence Checker": LiteralWrapper(nested_string_formatting(str(
                                         self.convergence_checker))),
                                     "Hunt Conditions": LiteralWrapper(nested_string_formatting(str(
                                         self.killing_conditions))),
                                     "Optimizer Selector": self.selector.glompo_log_repr(),
                                     "Max Parallel Optimizers": self.max_jobs},
                        "Counters": {"Function Evaluations": self.f_counter,
                                     "Hunts Started": self.hunt_counter,
                                     "Optimizers": {"Started": self.o_counter,
                                                    "Killed": len(self.hunt_victims),
                                                    "Converged": self.conv_counter}},
                        "Solution": {"fx": result.fx,
                                     "stats": result.stats,
                                     "origin": result.origin,
                                     "exit cond.": LiteralWrapper(nested_string_formatting(reason)),
                                     "x": result.x},
                        }
                self.logger.debug("Saving manager summary file.")
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)

            if self.summary_files == 3:
                self.logger.debug("Saving optimizer summary files.")
                self.opt_log.save("glompo_optimizer_logs")
            elif self.summary_files == 2 and best_id > 0:
                self.logger.debug("Saving optimizer summary file.")
                self.opt_log.save("glompo_best_optimizer_log", best_id)

    def _cleanup_crash(self, opt_reason: str):
        for opt_id in self.optimizer_packs:
            self.graveyard.add(opt_id)
            self.opt_log.put_metadata(opt_id, "Stop Time", datetime.now())
            self.opt_log.put_metadata(opt_id, "End Condition", opt_reason)
            self.optimizer_packs[opt_id].process.join(1)
            if self.optimizer_packs[opt_id].process.is_alive():
                self.optimizer_packs[opt_id].process.terminate()
                self.logger.debug(f"Termination signal sent to optimizer {opt_id}")
