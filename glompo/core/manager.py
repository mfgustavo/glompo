""" Contains GloMPOManager class which is the main user interface for GloMPO. """

import copy
import getpass
import glob
import logging
import multiprocessing as mp
import os
import queue
import shutil
import socket
import sys
import traceback
import warnings
from datetime import datetime
from time import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper as Dumper
try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False

from ._backends import CustomThread, ThreadPrintRedirect
from .optimizerlogger import OptimizerLogger
from ..common.customwarnings import NotImplementedWarning
from ..common.helpers import LiteralWrapper, literal_presenter, nested_string_formatting, unknown_object_presenter, \
    generator_presenter, optimizer_selector_presenter, mem_pprint
from ..common.namedtuples import Bound, OptimizerPackage, ProcessPackage, Result
from ..common.wrappers import process_print_redirect
from ..convergence import BaseChecker, KillsAfterConvergence
from ..generators import BaseGenerator, RandomGenerator
from ..hunters import BaseHunter
from ..opt_selectors.baseselector import BaseSelector
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("GloMPOManager",)


class GloMPOManager:
    """ Attempts to minimize a given function using numerous optimizers in parallel, based on their performance and
        decision criteria, will stop and intelligently restart others.
    """

    def __init__(self,
                 task: Callable[[Sequence[float]], float],
                 bounds: Sequence[Tuple[float, float]],
                 optimizer_selector: BaseSelector,
                 working_dir: str = ".",
                 overwrite_existing: bool = False,
                 max_jobs: Optional[int] = None,
                 backend: str = 'processes',
                 convergence_checker: Optional[BaseChecker] = None,
                 x0_generator: Optional[BaseGenerator] = None,
                 killing_conditions: Optional[BaseHunter] = None,
                 hunt_frequency: int = 100,
                 region_stability_check: bool = False,
                 report_statistics: bool = False,
                 summary_files: int = 0,
                 visualisation: bool = False,
                 visualisation_args: Optional[Dict[str, Any]] = None,
                 force_terminations_after: int = -1,
                 end_timeout: Optional[int] = None,
                 split_printstreams: bool = True):
        """
        Generates the environment for a globally managed parallel optimization job.

        Parameters
        ----------
        task: Callable[[Sequence[float]], float]
            Function to be minimized. Accepts a 1D sequence of parameter values and returns a single value.
            Note: Must be a standalone function which makes no modifications outside of itself.

        bounds: Sequence[Tuple[float, float]]
            Sequence of tuples of the form (min, max) limiting the range of each parameter. Do not use bounds to fix
            a parameter value as this will raise an error. Rather supply fixed parameter values through task_args or
            task_kwargs.

        optimizer_selector: BaseSelector
            Selection criteria for new optimizers, must be an instance of a BaseSelector subclass. BaseSelector
            subclasses are initialised by default with a set of BaseOptimizer subclasses the user would like to make
            available to the optimization. See BaseSelector and BaseOptimizer documentation for more details.

        working_dir: str = "."
            If provided, GloMPO wil redirect its outputs to the given directory.

        overwrite_existing: bool = False
            If True, GloMPO will overwrite existing files if any are found in the working_dir otherwise it will raise a
            FileExistsError if these results are detected.

        max_jobs: Optional[int] = None
            The maximum number of threads the manager may create. The number of threads created by a particular
            optimizer is given by optimizer.workers during its initialisation. An optimizer will not be started if
            the number of threads it creates will exceed max_jobs even if the manager is currently managing fewer
            than the number of jobs available. Defaults to one less than the number of CPUs available to the system.

        backend: str = 'processes'
            Indicates the form of parallelism used by the optimizers. 'processes' will bundle each optimizer into a
            multiprocessing.Process, while 'threads' will send the task to a threading.Thread.

            The appropriateness of each depends on the task itself. Using multiprocessing may provide computational
            advantages but becomes resource expensive as the task is duplicated between processes, there may also be
            I/O collisions if the task relies on external files during its calculation.

            If threads are used, make sure the task is thread-safe! Also note that forced terminations are not
            possible in this case and hanging optimizers will not be killed. The 'force_terminations_after' parameter
            is ignored.

            A third option is possible by sending 'processes_forced' but is **strongly discouraged**. In cases where two
            levels of parallelism exist (i.e. the optimizers and multiple parallel function evaluations therein). Then
            both levels can be configured to use processes to ensure adequate resource distribution by launching
            optimizers non-daemonically. By default the second parallelism level is threaded (see README for more
            details on this topic).

        convergence_checker: Optional[BaseChecker] = None
            Criteria used for convergence. A collection of subclasses of BaseChecker are provided, these can be
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
                E.g.: killing_conditions = (BestUnmoving(100, 0.01) & TimeAnnealing(2) & ValueAnnealing()) |
                                           ParameterDistance(0.1)
                In this case GloMPO will only allow a hunt to terminate an optimizer if
                    1) an optimizer's best value has not improved by more than 1% in 100 function calls,
                    2) and it fails an annealing type test based on how many iterations it has run,
                    3) and if fails an annealing type test based on how far the victim's value is from the best
                    optimizer's best value,
                    4) or the two optimizers are iterating very close to one another in parameter space
                Default (None): Killing is not used, i.e. the optimizer will not terminate optimizers.
            Note, for performance and to allow conditionality between hunters conditions are evaluated 'lazily' i.e.
            x or y will return if x is True without evaluating y. x and y will return False if x is False without
            evaluating y.

        hunt_frequency: int = 100
            The number of function calls between successive attempts to evaluate optimizer performance and determine
            if they should be terminated.

        region_stability_check: bool = False
            NotYetImplemented

        report_statistics: bool = False
            NotYetImplemented

        summary_files: int = 0
            Indicates the level of saving the user would like in terms of datafiles and plots:
                0 - No opt_log files are saved;
                1 - Only the manager summary file is saved;
                2 - The manager summary log and combined optimizers summary files are saved;
                3 - All of the above plus all the individual optimizer log files are saved;
                4 - All of the above plus plot of the optimizer trajectories
                5 - All of the above plus plots of trailed parameters as a function of optimizer iteration for each
                    optimizer.

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

        end_timeout: Optional[int] = None
            The amount of time the manager will wait trying to smoothly join each child optimizer at the end of the run.
            After this timeout, if the optimizer is still alive and a process, GloMPO will send a terminate signal to
            force it to close. However, threads cannot be terminated in this way and the manager can leave dangling
            threads at the end of its routine. Note that if the script ends after a GloMPO routine then all its children
            will be automatically garbage collected (provided processes_forced backend has not been used).

            By default, this timeout is 10s if a process backend is used and infinite of a threaded backend is used.
            This is the cleanest approach for threads but can cause very long wait times or deadlocks if the optimizer
            does not respond to close signals and does not converge.

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
        self._init_workdir = os.getcwd()
        if not isinstance(working_dir, str):
            warnings.warn(f"Cannot parse working_dir = {working_dir}. str or bytes expected. Using current "
                          f"work directory.", UserWarning)
            working_dir = "."
        self._workdir = working_dir

        # Save and wrap task
        if not callable(task):
            raise TypeError(f"{task} is not callable.")
        self.task = task
        self.logger.debug("Task wrapped successfully")

        # Save optimizer selection criteria
        if isinstance(optimizer_selector, BaseSelector):
            self.selector = optimizer_selector
        else:
            raise TypeError("optimizer_selector not an instance of a subclass of BaseSelector.")

        # Save bounds
        self.bounds: List[Bound] = []
        for bnd in bounds:
            if bnd[0] == bnd[1]:
                raise ValueError("Bounds min and max cannot be equal. Rather fix its value and remove it from the "
                                 "list of parameters. Fixed values can be supplied through task_args or task_kwargs.")
            if bnd[1] < bnd[0]:
                raise ValueError("Bound min cannot be larger than max.")
            self.bounds.append(Bound(bnd[0], bnd[1]))
        self.n_parms = len(self.bounds)

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
            self.logger.info("max_jobs set to one less than CPU count.")

        # Save convergence criteria
        if convergence_checker:
            if isinstance(convergence_checker, BaseChecker):
                self.convergence_checker = convergence_checker
            else:
                raise TypeError("convergence_checker not an instance of a subclass of BaseChecker.")
        else:
            self.convergence_checker = KillsAfterConvergence()
            self.logger.info("Convergence set to default: KillsAfterConvergence(0, 1)")

        # Save x0 generator
        if x0_generator:
            if isinstance(x0_generator, BaseGenerator):
                self.x0_generator = x0_generator
            else:
                raise TypeError("x0_generator not an instance of a subclass of BaseGenerator.")
        else:
            self.x0_generator = RandomGenerator(self.bounds)
            self.logger.info("x0 generator set to default: RandomGenerator()")

        # Save max conditions and counters
        self.result = Result(None, None, None, None)
        self.t_start = None
        self.dt_start = None
        self.converged = False
        self.o_counter = 0
        self.f_counter = 0
        self.last_hunt = 0
        self.conv_counter = 0  # Number of converged optimizers
        self.hunt_counter = 0  # Number of hunting jobs started
        self.hunt_victims: Dict[int, float] = {}  # opt_ids of killed jobs and timestamps when the signal was sent
        self.cpu_history = []
        self.mem_history = []
        self.load_history = []

        # Initialise system monitors
        self._process = psutil.Process()
        self._process.cpu_percent()  # First return is zero and must be ignored

        # Save behavioural args
        self.allow_forced_terminations = force_terminations_after > 0
        self._too_long = force_terminations_after
        self.region_stability_check = bool(region_stability_check)
        self.report_statistics = bool(report_statistics)
        self.summary_files = np.clip(int(summary_files), 0, 5)
        if self.summary_files != summary_files:
            self.logger.warning(f"summary_files argument given as {summary_files} clipped to {self.summary_files}")
        if summary_files:
            yaml.add_representer(LiteralWrapper, literal_presenter, Dumper=Dumper)
            yaml.add_multi_representer(BaseSelector, optimizer_selector_presenter, Dumper=Dumper)
            yaml.add_multi_representer(BaseGenerator, generator_presenter, Dumper=Dumper)
            yaml.add_multi_representer(object, unknown_object_presenter, Dumper=Dumper)
        self.split_printstreams = bool(split_printstreams)
        self.overwrite_existing = bool(overwrite_existing)
        self.visualisation = visualisation
        self.hunt_frequency = hunt_frequency
        self.spawning_opts = True
        self.opts_paused = False
        self.last_status = 0

        # Initialise support classes
        self.opt_log = OptimizerLogger()
        if visualisation:
            from .scope import GloMPOScope  # Only imported if needed to avoid matplotlib compatibility issues
            self.scope = GloMPOScope(**visualisation_args) if visualisation_args else GloMPOScope()
        if region_stability_check:
            warnings.warn("region_stability_check not implemented. Ignoring.", NotImplementedWarning)
            self.logger.warning("region_stability_check not yet implemented")
        if report_statistics:
            warnings.warn("report_statistics not implemented. Ignoring.", NotImplementedWarning)
            self.logger.warning("report_statistics not yet implemented")

        # Save killing conditions
        if killing_conditions:
            if isinstance(killing_conditions, BaseHunter):
                self.killing_conditions = killing_conditions
            else:
                raise TypeError("killing_conditions not an instance of a subclass of BaseHunter.")
        else:
            self.killing_conditions = None
            self.logger.info("Hunting will not be used by the manager.")

        # Setup backend
        if any([backend == valid_opt for valid_opt in ('processes', 'threads', 'processes_forced')]):
            self._proc_backend = 'processes' in backend
            self.opts_daemonic = backend != 'processes_forced'
        else:
            self._proc_backend = True
            self.opts_daemonic = True
            self.logger.warning(f"Unable to parse backend '{backend}'. 'processes' or 'threads' expected."
                                f"Defaulting to 'processes'.")
            warnings.warn(f"Unable to parse backend '{backend}'. 'processes' or 'threads' expected."
                          f"Defaulting to 'processes'.")
        self.optimizer_packs: Dict[int, ProcessPackage] = {}
        self.graveyard: Set[int] = set()
        self.last_feedback: Dict[int, float] = {}

        self._mp_manager = mp.Manager()
        self.optimizer_queue = self._mp_manager.Queue()

        if self.split_printstreams and not self._proc_backend:
            sys.stdout = ThreadPrintRedirect(sys.stdout)
            sys.stderr = ThreadPrintRedirect(sys.stderr)

        if self.allow_forced_terminations and not self._proc_backend:
            warnings.warn("Cannot use force terminations with threading.", UserWarning)
            self.logger.warning("Cannot use force terminations with threading.")

        if end_timeout:
            self.end_timeout = end_timeout
        else:
            if self._proc_backend:
                self.end_timeout = 10
            else:
                self.end_timeout = None

        self.logger.info("Initialization Done")

    def start_manager(self) -> Result:
        """ Begins the optimization routine and returns the selected minimum in an instance of MinimizeResult. """

        reason = "None"
        caught_exception = None
        best_id = -1

        # Move into or make working dir
        os.makedirs(self._workdir, exist_ok=True)
        os.chdir(self._workdir)

        # Purge Old Results
        files = os.listdir()
        if any([file in files for file in ["glompo_manager_log.yml", "glompo_optimizer_logs"]]):
            if self.overwrite_existing:
                self.logger.debug("Old results found")
                to_remove = ["glompo_manager_log.yml", "opt_best_summary.yml"]
                to_remove += glob.glob("trajectories*.png", recursive=False)
                to_remove += glob.glob("opt*_parms.png", recursive=False)
                for old in to_remove:
                    try:
                        os.remove(old)
                    except FileNotFoundError:
                        continue
                shutil.rmtree("glompo_optimizer_logs", ignore_errors=True)
                shutil.rmtree("glompo_optimizer_printstreams", ignore_errors=True)
                self.logger.warning("Deleted old results.")
            else:
                raise FileExistsError(
                    "Previous results found. Remove, move or rename them. Alternatively, select "
                    "another working_dir or set overwrite_existing=True.")

        if self.visualisation and self.scope.record_movie:
            self.scope.setup_moviemaker()  # Declared here to ensure no overwriting and creation in correct dir

        if self.split_printstreams:
            os.makedirs("glompo_optimizer_printstreams", exist_ok=True)

        try:
            self.logger.info("Starting GloMPO Optimization Routine")

            self.t_start = time()
            self.last_status = self.t_start
            self.dt_start = datetime.now()

            while not self.converged:

                self.logger.debug("Checking for available optimizer slots")
                self._fill_optimizer_slots()
                self.logger.debug("New optimizer check done")

                self.logger.debug("Checking optimizer signals")
                for opt_id in self.optimizer_packs:
                    self._check_signals(opt_id)
                self.logger.debug("Signal check done.")

                self.logger.debug("Checking optimizer iteration results")
                self._process_results()
                self.logger.debug("Iteration results check done.")

                self.result = self._find_best_result()
                if self.result.origin and 'opt_id' in self.result.origin:
                    best_id = self.result.origin['opt_id']

                if best_id > 0 and self.killing_conditions:
                    self._start_hunt(best_id)

                self.logger.debug("Checking for user interventions.")
                self._is_manual_shutdowns()

                self.logger.debug("Checking for hanging processes")
                self._inspect_children()

                all_dead = len([p for p in self.optimizer_packs.values() if p.process.is_alive()]) == 0
                checker_condition = self.convergence_checker(self)

                if checker_condition:
                    reason = self.convergence_checker.str_with_result()
                else:
                    reason = "No optimizers alive, spawning stopped."

                self.converged = checker_condition or (all_dead and not self.spawning_opts)
                if self.converged:
                    self.logger.info("Convergence Reached")

                if time() - self.last_status > 6:
                    self.last_status = time()
                    processes = [pack.slots for pack in self.optimizer_packs.values() if pack.process.is_alive()]
                    f_best = f'{self.result.fx:.3E}' if self.result.fx is not None else None
                    evals = f"{self.f_counter:,}".replace(',', ' ')
                    status_mess = f"Status: \n" \
                                  f"    {'Time Elapsed:':.<26} {datetime.now() - self.dt_start}\n" \
                                  f"    {'Optimizers Alive:':.<26} {len(processes)}\n" \
                                  f"    {'Slots Filled:':.<26} {sum(processes)}/{self.max_jobs}\n" \
                                  f"    {'Function Evaluations:':.<26} {evals}\n" \
                                  f"    Current Optimizer f_vals:\n"
                    for opt_id in self.optimizer_packs:
                        if self.optimizer_packs[opt_id].process.is_alive():
                            hist = self.opt_log.get_history(opt_id, 'fx')
                            if len(hist) > 0:
                                width = 21 if hist[-1] < 0 else 22
                                status_mess += f"        {f'Optimizer {opt_id}':.<{width}} {hist[-1]:.3E}\n"
                    status_mess += f"    {'Overall f_best:':.<25} {f_best}\n"
                    if HAS_PSUTIL:
                        with self._process.oneshot():
                            self.cpu_history.append(self._process.cpu_percent())
                            self.mem_history.append(self._process.memory_full_info().uss)
                        self.load_history.append(psutil.getloadavg())

                        status_mess += f"    {'CPU Usage:':.<26} {self.cpu_history[-1]}%\n"
                        status_mess += f"    {'Virtual Memory:':.<26} {mem_pprint(self.mem_history[-1])}\n"
                        try:
                            status_mess += f"    {'System Load:':.<26} {self.load_history[-1]}\n"
                        except AttributeError:
                            pass
                    self.logger.info(status_mess)

            self.logger.info("Exiting manager loop")
            self.logger.info(f"Exit conditions met: \n"
                             f"{nested_string_formatting(reason)}")

            self.logger.debug("Cleaning up multiprocessing")
            self._stop_all_children()

        except KeyboardInterrupt:
            caught_exception = "User Interrupt"
            self.logger.error("Caught User Interrupt, closing GloMPO gracefully.")
            warnings.warn("Optimization failed. Caught User Interrupt", RuntimeWarning)
            self._cleanup_crash("User Interrupt")

        except Exception as e:
            caught_exception = "".join(traceback.TracebackException.from_exception(e).format())
            self.logger.critical("Critical error encountered. Attempting to close GloMPO gracefully", exc_info=e)
            warnings.warn(f"Optimization failed. Caught exception: {e}", RuntimeWarning)
            self._cleanup_crash("GloMPO Crash")

        finally:

            self.logger.info("Cleaning up and closing GloMPO")

            if self.visualisation:
                if self.scope.record_movie and not caught_exception:
                    self.logger.debug("Generating movie")
                    self.scope.generate_movie()
                self.scope.close_fig()

            self.logger.debug("Saving summary file results")
            self._save_log(self.result, reason, caught_exception)

            self.result = Result(self.result.x,
                                 self.result.fx,
                                 {**self.result.stats, 'end_cond': reason},
                                 self.result.origin)

            if not self._proc_backend and self.split_printstreams:
                sys.stdout.close()
                sys.stderr.close()

            os.chdir(self._init_workdir)

            self.logger.info("GloMPO Optimization Routine Done")

            return self.result

    def _fill_optimizer_slots(self):
        """ Starts new optimizers if there are slots available. """
        processes = [pack.slots for pack in self.optimizer_packs.values() if pack.process.is_alive()]
        count = sum(processes)

        is_possible = True  # Flag if no optimizer can fit in the slots available due to its configuration
        started_new = False
        while count < self.max_jobs and is_possible and self.spawning_opts:
            opt = self._setup_new_optimizer(self.max_jobs - count)
            if opt:
                self._start_new_job(*opt)
                count += opt.slots
                started_new = True
            else:
                is_possible = False

        if started_new:
            processes = [pack.slots for pack in self.optimizer_packs.values() if pack.process.is_alive()]
            f_best = f'{self.result.fx:.3E}' if self.result.fx is not None else None
            self.logger.info(f"Status: {len(processes)} optimizers alive, {sum(processes)}/{self.max_jobs} slots filled"
                             f", {self.f_counter} function evaluations, f_best = {f_best}.")

    def _start_new_job(self, opt_id: int, optimizer: BaseOptimizer, call_kwargs: Dict[str, Any],
                       pipe: mp.connection.Connection, event: mp.Event, workers: int):
        """ Given an initialised optimizer and multiprocessing variables, this method packages them and starts a new
            process.
        """

        self.logger.info(f"Starting Optimizer: {opt_id}")

        task = self.task
        x0 = self.x0_generator.generate(self)
        bounds = np.array(self.bounds)
        # noinspection PyProtectedMember
        target = optimizer._minimize

        if self.split_printstreams and self._proc_backend:
            # noinspection PyProtectedMember
            target = process_print_redirect(opt_id, optimizer._minimize)

        kwargs = {'target': target,
                  'args': (task, x0, bounds),
                  'kwargs': call_kwargs,
                  'name': f"Opt{opt_id}",
                  'daemon': self.opts_daemonic}
        if self._proc_backend:
            process = mp.Process(**kwargs)
        else:
            process = CustomThread(redirect_print=self.split_printstreams, **kwargs)

        self.optimizer_packs[opt_id] = ProcessPackage(process, pipe, event, workers)
        self.optimizer_packs[opt_id].process.start()
        self.last_feedback[opt_id] = time()

        if self.visualisation:
            if opt_id not in self.scope.streams:
                self.scope.add_stream(opt_id, type(optimizer).__name__)

    def _setup_new_optimizer(self, slots_available: int) -> OptimizerPackage:
        """ Selects and initializes new optimizer and multiprocessing variables. Returns an OptimizerPackage which
            can be sent to _start_new_job to begin new process.
        """

        selector_return = self.selector.select_optimizer(self, self.opt_log, slots_available)

        if not selector_return:
            if selector_return is False:
                self.logger.info("Optimizer spawning deactivated.")
                self.spawning_opts = False
            return None

        selected, init_kwargs, call_kwargs = selector_return
        if not self._proc_backend:
            # Callbacks need to be copied in the case of threaded backends because otherwise they will behave
            # globally rather than on individual optimizers as expected. All kwargs are copied in this way to prevent
            # any strange race conditions and multithreading artifacts.
            init_kwargs = copy.deepcopy(init_kwargs)
            call_kwargs = copy.deepcopy(call_kwargs)
        self.o_counter += 1

        self.logger.info(f"Setting up optimizer {self.o_counter} of type {selected.__name__}")

        parent_pipe, child_pipe = mp.Pipe()
        event = self._mp_manager.Event()
        event.set()

        if 'backend' in init_kwargs:
            backend = init_kwargs['backend']
            del init_kwargs['backend']
        else:
            backend = 'threads' if self.opts_daemonic else 'processes'

        optimizer = selected(opt_id=self.o_counter,
                             signal_pipe=child_pipe,
                             results_queue=self.optimizer_queue,
                             pause_flag=event,
                             backend=backend,
                             **init_kwargs)

        self.opt_log.add_optimizer(self.o_counter, type(optimizer).__name__, datetime.now())

        if call_kwargs:
            return OptimizerPackage(self.o_counter, optimizer, call_kwargs, parent_pipe, event, init_kwargs['workers'])
        return OptimizerPackage(self.o_counter, optimizer, {}, parent_pipe, event, init_kwargs['workers'])

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

            # Find dead optimizer processes that did not properly signal their termination.
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
                    time() - self.last_feedback[opt_id] > self._too_long and \
                    self.allow_forced_terminations and \
                    opt_id not in self.hunt_victims and \
                    self._proc_backend:
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
                    time() - self.hunt_victims[opt_id] > self._too_long and \
                    self._proc_backend:
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

        # Pause / restart optimizers based on the status of the queue
        if self.optimizer_queue.qsize() > 10 and not self.opts_paused:
            self.logger.debug(f"Results queue swamped ({self.optimizer_queue.qsize()} results). Pausing optimizers.")
            self._toggle_optimizers(0)
        elif self.optimizer_queue.qsize() <= 10 and self.opts_paused:
            self.logger.debug("Resuming optimizers.")
            self._toggle_optimizers(1)

        for i in range(10):
            try:
                res = self.optimizer_queue.get(block=True, timeout=1)
            except queue.Empty:
                self.logger.debug("Timeout on result queue.")
                break

            self.last_feedback[res.opt_id] = time()
            self.f_counter += res.i_fcalls

            history = self.opt_log.get_history(res.opt_id, "f_call_opt")
            if len(history) > 0:
                opt_fcalls = history[-1] + res.i_fcalls
            else:
                opt_fcalls = res.i_fcalls

            if res.opt_id not in self.hunt_victims:
                self.opt_log.put_iteration(res.opt_id, res.n_iter, self.f_counter, opt_fcalls, list(res.x), res.fx)
                self.logger.debug(f"Result from {res.opt_id} @ iter {res.n_iter} fx = {res.fx}")

                if self.visualisation:
                    self.scope.update_optimizer(res.opt_id, (self.f_counter, res.fx))
                    if res.final:
                        self.scope.update_norm_terminate(res.opt_id)

    def _start_hunt(self, hunter_id: int):
        """ Creates a new hunt with the provided hunter_id as the 'best' optimizer looking to terminate
            the other active optimizers according to the provided killing_conditions.
        """

        if self.f_counter - self.last_hunt > self.hunt_frequency:
            self.hunt_counter += 1
            self.last_hunt = self.f_counter

            self.logger.debug("Starting hunt")
            for victim_id in self.optimizer_packs:
                in_graveyard = victim_id in self.graveyard
                has_points = len(self.opt_log.get_history(victim_id, "fx")) > 0
                if not in_graveyard and has_points and victim_id != hunter_id:
                    self.logger.debug(f"Optimizer {hunter_id} -> Optimizer {victim_id} hunt started.")
                    kill = self.killing_conditions(self.opt_log, hunter_id, victim_id)

                    if kill:
                        reason = nested_string_formatting(self.killing_conditions.str_with_result())
                        self.logger.info(f"Optimizer {hunter_id} wants to kill Optimizer {victim_id}:\n"
                                         f"Reason:\n"
                                         f"{reason}")

                        if victim_id not in self.graveyard:
                            self._shutdown_job(victim_id, hunter_id, reason)

            self.logger.debug("Hunting complete")

    def _is_manual_shutdowns(self):
        files = os.listdir()
        files = [file for file in files if "STOP_" in file]
        for file in files:
            try:
                _, opt_id = file.split('_')
                opt_id = int(opt_id)
                if opt_id not in self.graveyard:
                    self._shutdown_job(opt_id, None, "User STOP file intervention.")
                    self.logger.info(f"STOP file found for Optimizer {opt_id}")
                    os.remove(file)
            except ValueError as e:
                self.logger.debug("Error encountered trying to process STOP files.", exc_info=e)
                continue

    def _shutdown_job(self, opt_id: int, hunter_id: int, reason: str):
        """ Sends a stop signal to optimizer opt_id and updates variables around its termination. """
        self.hunt_victims[opt_id] = time()
        self.graveyard.add(opt_id)

        self.optimizer_packs[opt_id].signal_pipe.send(1)
        self.logger.debug(f"Termination signal sent to {opt_id}")

        self.opt_log.put_metadata(opt_id, "Stop Time", datetime.now())
        self.opt_log.put_metadata(opt_id, "End Condition", LiteralWrapper(f"GloMPO Termination\n"
                                                                          f"Hunter: {hunter_id}\n"
                                                                          f"Reason: \n"
                                                                          f"{reason}"))

        if self.visualisation:
            self.scope.update_kill(opt_id)

    def _find_best_result(self) -> Result:
        best_fx = np.inf
        best_x = []
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

        best_stats = {'f_evals': self.f_counter,
                      'opts_started': self.o_counter,
                      'opts_killed': len(self.hunt_victims),
                      'opts_conv': self.conv_counter,
                      'end_cond': None}

        return Result(best_x, best_fx, best_stats, best_origin)

    def _stop_all_children(self):
        if self.opts_paused:
            self._toggle_optimizers(1)

        for opt_id in self.optimizer_packs:
            if self.optimizer_packs[opt_id].process.is_alive():
                self.optimizer_packs[opt_id].signal_pipe.send(1)
                self.graveyard.add(opt_id)
                self.opt_log.put_metadata(opt_id, "Stop Time", datetime.now())
                self.opt_log.put_metadata(opt_id, "End Condition", "GloMPO Convergence")
                self.optimizer_packs[opt_id].process.join(self.end_timeout)
                if self.optimizer_packs[opt_id].process.is_alive():
                    if self._proc_backend:
                        self.logger.info(f"Termination signal sent to optimizer {opt_id}")
                        self.optimizer_packs[opt_id].process.terminate()
                    else:
                        self.logger.warning(f"Optimizer {opt_id} still alive after join attempt. GloMPO will end while "
                                            f"it is alive. Terminations cannot be sent to threads.")

    def _save_log(self, result: Result, reason: str, caught_exception: bool):
        if self.summary_files > 0:
            if caught_exception:
                reason = f"Process Crash: {caught_exception}"
            with open("glompo_manager_log.yml", "w") as file:

                # Requires updated psutil
                if HAS_PSUTIL:
                    try:
                        run_info = {"Max Memory": mem_pprint(np.max(self.mem_history)),
                                    "CPU": {"Cores Available": len(self._process.cpu_affinity()),
                                            "Frequency": f"{psutil.cpu_freq().max / 1000}GHz",
                                            "Load": {
                                                "Average": list(np.round(np.average(self.load_history, axis=0), 3)),
                                                "Std. Dev.": list(np.round(np.std(self.load_history, axis=0), 3))},
                                            "CPU Usage(%)": {"Average": np.round(np.average(self.cpu_history), 2),
                                                             "Std. Dev.": np.round(np.std(self.cpu_history), 2)}}}
                    except AttributeError:
                        run_info = "<COULD NOT MEASURE. REQUIRES psutil>=5 UPDATE PSUTIL>"

                data = {"Assignment": {"Task": type(self.task).__name__,
                                       "Working Dir": os.getcwd(),
                                       "Username": getpass.getuser(),
                                       "Hostname": socket.gethostname(),
                                       "Start Time": self.dt_start,
                                       "Stop Time": datetime.now()},
                        "Settings": {"x0 Generator": self.x0_generator,
                                     "Convergence Checker": LiteralWrapper(nested_string_formatting(str(
                                         self.convergence_checker))),
                                     "Hunt Conditions": LiteralWrapper(nested_string_formatting(str(
                                         self.killing_conditions))) if self.killing_conditions else
                                     self.killing_conditions,
                                     "Optimizer Selector": self.selector,
                                     "Max Jobs": self.max_jobs},
                        "Counters": {"Function Evaluations": self.f_counter,
                                     "Hunts Started": self.hunt_counter,
                                     "Optimizers": {"Started": self.o_counter,
                                                    "Killed": len(self.hunt_victims),
                                                    "Converged": self.conv_counter}},
                        "Run Information": run_info,
                        "Solution": {"fx": result.fx,
                                     "origin": result.origin,
                                     "exit cond.": LiteralWrapper(nested_string_formatting(reason)),
                                     "x": result.x},
                        }
                self.logger.debug("Saving manager summary file.")
                yaml.dump(data, file, Dumper=Dumper, default_flow_style=False, sort_keys=False)

            if self.summary_files >= 2:
                self.logger.debug("Saving optimizers summary file.")
                self.opt_log.save_summary("opt_best_summary.yml")
            if self.summary_files >= 3:
                self.logger.debug("Saving optimizer log files.")
                self.opt_log.save_optimizer("glompo_optimizer_logs")
            if self.summary_files >= 4:
                self.logger.debug("Saving trajectory plot.")
                signs = set()
                large = 0
                small = float('inf')
                for opt_id in self.optimizer_packs:
                    fx = self.opt_log.get_history(opt_id, 'fx')
                    [signs.add(i) for i in set(np.sign(fx))]
                    if len(fx) > 0:
                        large = max(fx) if max(fx) > large else large
                        small = min(fx) if min(fx) < small else small

                all_sign = len(signs) == 1
                range_large = large - small > 1e5
                log_scale = all_sign and range_large
                for best_fx in (True, False):
                    name = "trajectories_"
                    name += "log_" if log_scale else ""
                    name += "best_" if best_fx else ""
                    name = name[:-1] if name.endswith("_") else name
                    name += ".png"
                    self.opt_log.plot_trajectory(name, log_scale, best_fx)
            if self.summary_files == 5:
                self.logger.debug("Saving optimizer parameter trials.")
                self.opt_log.plot_optimizer_trials()

    def _cleanup_crash(self, opt_reason: str):
        for opt_id in self.optimizer_packs:
            try:
                self.opt_log.get_metadata(opt_id, "End Condition")
                self.opt_log.get_metadata(opt_id, "Stop Time")
            except KeyError:
                self.graveyard.add(opt_id)
                self.opt_log.put_metadata(opt_id, "Stop Time", datetime.now())
                self.opt_log.put_metadata(opt_id, "End Condition", opt_reason)
            self.optimizer_packs[opt_id].process.join(10)
            if self.optimizer_packs[opt_id].process.is_alive():
                if self._proc_backend:
                    self.optimizer_packs[opt_id].process.terminate()
                    self.logger.debug(f"Termination signal sent to optimizer {opt_id}")
                else:
                    self.logger.warning(f"Could not join optimizer {opt_id}. May crash out with it still running and "
                                        f"thus generate errors.")

    def _toggle_optimizers(self, on_off: int):
        """ Sends pause or resume signals to all optimizers based on the on_off parameter:
            0 -> Optimizers off
            1 -> Optimizers on
        """
        self.opts_paused = not on_off
        for pack in self.optimizer_packs.values():
            if pack.process.is_alive():
                if on_off == 1:
                    pack.allow_run_event.set()
                else:
                    pack.allow_run_event.clear()
