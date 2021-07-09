""" Contains GloMPOManager class which is the main user interface for GloMPO. """

import copy
import getpass
import logging
import multiprocessing as mp
import queue
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from pickle import PickleError
from time import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import yaml

try:
    import dill

    HAS_DILL = True
except ModuleNotFoundError:
    HAS_DILL = False

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

try:
    import psutil

    HAS_PSUTIL = psutil.version_info >= (5, 6, 2)
except ModuleNotFoundError:
    HAS_PSUTIL = False

from ._backends import CustomThread, ThreadPrintRedirect
from .optimizerlogger import OptimizerLogger
from ..common.helpers import LiteralWrapper, literal_presenter, nested_string_formatting, \
    unknown_object_presenter, generator_presenter, optimizer_selector_presenter, present_memory, FlowList, \
    flow_presenter, numpy_array_presenter, numpy_dtype_presenter, BoundGroup, bound_group_presenter, \
    CheckpointingError, is_bounds_valid
from ..common.namedtuples import Bound, IterationResult, OptimizerPackage, ProcessPackage, Result, OptimizerCheckpoint
from ..common.wrappers import process_print_redirect
from ..convergence import BaseChecker, KillsAfterConvergence
from ..generators import BaseGenerator, RandomGenerator
from ..hunters import BaseHunter
from ..opt_selectors.baseselector import BaseSelector
from ..optimizers.baseoptimizer import BaseOptimizer
from .checkpointing import CheckpointingControl

__all__ = ("GloMPOManager",)


class GloMPOManager:
    """ Attempts to minimize a given function using numerous optimizers in parallel, based on their performance and
        decision criteria, will stop and intelligently restart others.

        Methods
        -------
        new_manager
            Class method. Configure a new manager object. Equivalent to GloMPOManager().setup()
        load_manager
            Class method. Extract manager from a GloMPO checkpoint file. Equivalent to GloMPOManager().load_checkpoint
        setup
            Configure a new optimization management with the arguments provided.
        load_checkpoint
            Extract optimization state from a GloMPO checkpoint.
        start_manager
            Start the managed optimization.
        checkpoint
            Create a GloMPO checkpoint file preserving the state of the manager.
        dump_state
            Produce GloMPO log files from the current manager state.

        Attributes
        ----------
        allow_forced_terminations: bool
            True if the manager is allowed to force terminate optimizers which appear non-responsive (i.e. do not
            provide feedback within a specified period of time.
        bounds: Sequence[Bound]
            (Min, max) tuples for each parameter being optimized beyond which optimizers will not explore.
        checkpoint_control: CheckpointingControl
            GloMPO object containing all checkpointing settings if this feature is being used.
            (See glompo.core.checkpointing.CheckpointingControl)
        checkpoint_history: Set[str]
            Set of names of checkpoints constructed by the manager.
        converged: bool
            True if the conditions of convergence_checker have been met.
        conv_counter: int
            Count of the number of optimizers which converged according to their own configuration (as opposed to being
            terminated by the manager).
        convergence_checker: BaseChecker
            GloMPO object which evaluates whether conditions are met for overall manager termination.
            (See glompo.convergence.basechecker.BaseChecker).
        cpu_history: List[float]
            History of CPU percentage usage snapshots (taken every status_frequency seconds). This is the CPU percentage
            used only by the process and its children not the load on the whole system.
        dt_ends: List[datetime]
            List of datetime objects recording the end of each optimization session for a problem optimized through
            several checkpoints.
        dt_starts: List[datetime]
            List of datetime objects recording the start of each optimization session for a problem optimized through
            several checkpoints.
        end_timeout: float
            Amount of time the manager will wait to join child processes before forcibly terminating them (if children
            are processes) or allowing them to eventually crash out themselves (if children are threads). The latter is
            not recommended as essentially these threads can become orphaned and continue to use resources in the
            background. Unfortunately, threads cannot be forcibly terminated.
        f_counter: int
            Number of times the optimization task has been evaluated.
        hunt_counter: int
            Count of the number of times the manager has evaluated killing_conditions in an attempt to terminate one of
            its children.
        hunt_frequency: int
            Frequency (in terms of number of function evaluations) between manager 'hunts' (i.e. evaluation of
            killing_conditions in an attempt to terminate children.
        hunt_victims: Dict[int, float]
            Mapping of manager-killed optimizer ID numbers and timestamps when they were terminated.
        incumbent_sharing: bool
            If True the manager will send iteration information about the best ever seen solution to all its children
            whenever this is updated.
        killing_conditions: BaseHunter
            GloMPO object which evaluates whether an optimizer meets its conditions to be terminated early.
            (See glompo.hunters.basehunter.BaseHunter).
        last_hunt: int
            Evaluation number at which the last hunt was executed.
        last_iter_checkpoint: int
            f_counter of last attempted checkpoint (regardless of success or failure)
        last_opt_spawn: Tuple[int, int]
            Tuple of f_counter and o_counter at which the last child optimizer was started.
        last_status: float
            Timestamp when the last logging status message was printed.
        last_time_checkpoint: float
            Timestamp of last attempted checkpoint (regardless of success or failure)
        load_history: List[Tuple[float, float, float]]
            History of system load snapshots (taken every status_frequency seconds). This is is a system wide value, not
            tied to the specific process.
        logger: logging.Logger
            GloMPO has built-in logging to allowed tracking during an optimization. See README for more details. This
            attribute accesses the manager logger object.
        max_jobs: int
            Maximum number of calculation `slots` used by all the child optimizers. This generally equates to the number
            of processing cores available which the child optimizers may fill with threads or processes depending on
            their configuration. Alternativly, each child optimizer may work serially and take one of these slots.
        mem_history: List[float]
            History of memory usage snapshots (taken every status_frequency seconds). Details memeory used by the
            process and its children.
        n_parms: int
            Dimensionality of the optimization problem.
        o_counter: int
            Number of optimizers started.
        opt_crashed: bool
            True if any child optimizer crashed during its execution.
        opt_log: OptimizerLogger
            GloMPO object collecting the entire iteration history and metadata of the manager's children.
            (See glompo.core.optimizerlogger.OptimizerLogger)
        opt_selector: BaseSelector
            Object which returns an optimizer class and its configuration when requested by the manager. Can be based on
            previous results delivered by other optimizers. (See glompo.opt_selectors.baseselector.BaseSelector)
        optimizer_queue: queue.Queue
            Common concurrency tool into which all results are paced by child optimizers.
        opts_daemonic: bool
            True if manager children are spawned as daemons. Default is True but can be set to False if double process
            layers are needed (see README for details on this).
        overwrite_existing: bool
            True if any old files detected in the working directory maybe be deleted when the optimization run begins.
        proc_backend: bool
            True if the manager children are spawned as processes, False if they are spawned as threads.
        result: Result
            Incumbent best solution found by any child optimizer.
        scope: Optional['GloMPOScope']
            GloMPO object presenting the optimization graphically in real time.
            (See glompo.core.scope.GloMPOScope).
        split_printstreams: bool
            True if the printstreams for children are redirected to individual files.
        spawning_opts: bool
            True if the manager is allowed to create new children. The manager will shutdown if all children terminate
            and this is False.
        status_frequency: float
            Frequency (in seconds) with which a status message is produced for the logger.
        summary_files: int
            Indicates the level of saving the user would like in terms of datafiles and plots:
                0 - No opt_log files are saved;
                1 - Only the manager summary file is saved;
                2 - The manager summary log and combined optimizers summary files are saved;
                3 - All of the above plus all the individual optimizer log files are saved;
                4 - All of the above plus plot of the optimizer trajectories
                5 - All of the above plus plots of trailed parameters as a function of optimizer iteration for each
                    optimizer.
        t_end: float
            Timestamp of the ending time of an optimization run.
        t_start: float
            Timestamp of the starting time of an optimization run.
        t_used: float
            Total time in seconds used by PREVIOUS optimization runs. This will be zero unless the manager has been
            loaded from a checkpoint.
        task: Callable[[Sequence[float]], float]
            Function being minimize by the optimizers.
        visualisation: bool
            True if the optimization is presented graphically in real time using glompo.core.scope.GloMPOScope.
        visualisation_args: Dict[str, Any]
            Configuration arguments used for glompo.core.scope.GloMPOScope if the optimization is being visualised
            dynamically.
        working_dir: Path
            Working directory in which all output files and directories are created. Note, the manager does not change
            the current working directory during the run.
        x0_generator: BaseGenerator
            GloMPO object which returns a starting location for a new child optimizer. Can be based on previous results
            delivered by other optimizers. (See glompo.generators.basegenerator.BaseGenerator).
    """

    @classmethod
    def new_manager(cls, *args, **kwargs) -> 'GloMPOManager':
        """ Class method wrapper around GloMPOManager.setup method to directly initialise a new manager instance.

            See Also
            --------
            GloMPOManager.setup for argument specs
        """

        manager = cls()
        manager.setup(*args, **kwargs)
        return manager

    @classmethod
    def load_manager(cls, *args, **kwargs) -> 'GloMPOManager':
        """ Class method wrapper around GloMPOManager.load_checkpoint to directly initialise a manager from a
            checkpoint.

            See Also
            --------
            GloMPOManager.load_checkpoint for argument specs
        """

        manager = cls()
        manager.load_checkpoint(*args, **kwargs)
        return manager

    # noinspection PyTypeChecker
    def __init__(self):
        """ The manager is not initialised directly. Either use GloMPOManager().setup() to build a new optimization
            or GloMPOManager.load_checkpoint() to resume an optimization from a previously saved checkpoint file.
            Alternatively, class methods GloMPOManager.new_manager and GloMPOManager.load_manager are also provided.
        """

        # Filter Warnings
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("always", RuntimeWarning)

        self._is_restart: bool = None

        self.logger = logging.getLogger('glompo.manager')
        self.working_dir: Path = None

        self._mp_manager = mp.Manager()
        self.optimizer_queue = self._mp_manager.Queue(10)

        yaml.add_representer(LiteralWrapper, literal_presenter, Dumper=Dumper)
        yaml.add_representer(FlowList, flow_presenter, Dumper=Dumper)
        yaml.add_representer(np.ndarray, numpy_array_presenter, Dumper=Dumper)
        yaml.add_representer(BoundGroup, bound_group_presenter, Dumper=Dumper)
        yaml.add_multi_representer(np.generic, numpy_dtype_presenter, Dumper=Dumper)
        yaml.add_multi_representer(BaseSelector, optimizer_selector_presenter, Dumper=Dumper)
        yaml.add_multi_representer(BaseGenerator, generator_presenter, Dumper=Dumper)
        yaml.add_multi_representer(object, unknown_object_presenter, Dumper=Dumper)

        self.task: Callable[[Sequence[float]], float] = None
        self.opt_selector: BaseSelector = None
        self.bounds: Sequence[Bound] = None
        self.n_parms: int = None
        self.max_jobs: int = None
        self.convergence_checker: BaseChecker = None
        self.x0_generator: BaseGenerator = None
        self.killing_conditions: BaseHunter = None

        self.result = Result(None, None, None, None)
        self.t_start: float = None  # Session start time
        self.t_end: float = None  # Session end time
        self.t_used: float = 0  # Time used during previous sessions if loading from checkpoint
        self.dt_starts: List[datetime] = []
        self.dt_ends: List[datetime] = []
        self.converged: bool = None
        self.opt_crashed: bool = None
        self.end_timeout: float = None
        self.o_counter = 0
        self.f_counter = 0
        self.last_hunt = 0
        self.conv_counter = 0
        self.hunt_counter = 0
        self.last_status = 0
        self.last_opt_spawn = (0, 0)
        self.last_time_checkpoint = 0
        self.last_iter_checkpoint = 0
        self.checkpoint_history: Set[str] = set()

        self._process: Optional['psutil.Process'] = None
        self.cpu_history: List[float] = []
        self.mem_history: List[float] = []
        self.load_history: List[Tuple[float, float, float]] = []

        self.hunt_victims: Dict[int, float] = {}  # opt_ids of killed jobs and timestamps when the signal was sent
        self._optimizer_packs: Dict[int, ProcessPackage] = {}  # Dictionary of living or recently living optimizers.
        self._graveyard: Set[int] = set()
        self._last_feedback: Dict[int, float] = {}
        self._opt_checkpoints: Dict[int, OptimizerCheckpoint] = {}  # Type & slots of every opt for checkpt loading

        self.allow_forced_terminations: bool = None
        self._too_long: float = None
        self.summary_files: int = None
        self.split_printstreams: bool = None
        self.overwrite_existing: bool = None
        self.visualisation: bool = None
        self.visualisation_args: Dict[str, Any] = {}
        self.hunt_frequency: int = None
        self.spawning_opts: bool = None
        self.incumbent_sharing: bool = None
        self.status_frequency: float = None
        self.checkpoint_control: CheckpointingControl = None

        self.opt_log: OptimizerLogger = OptimizerLogger()
        # noinspection PyUnresolvedReferences
        self.scope: Optional['GloMPOScope'] = None

        self.proc_backend: bool = None
        self.opts_daemonic: bool = None

    @property
    def is_initialised(self) -> bool:
        """ Returns True if this GloMPOManager instance has been initialised. Multiple initialisations are not allowed.
        """

        return self._is_restart is not None

    def setup(self,
              task: Callable[[Sequence[float]], float],
              bounds: Sequence[Tuple[float, float]],
              opt_selector: BaseSelector,
              working_dir: Union[Path, str] = ".",
              overwrite_existing: bool = False,
              max_jobs: Optional[int] = None,
              backend: str = 'processes',
              convergence_checker: Optional[BaseChecker] = None,
              x0_generator: Optional[BaseGenerator] = None,
              killing_conditions: Optional[BaseHunter] = None,
              share_best_solutions: bool = False,
              hunt_frequency: int = 100,
              status_frequency: int = 600,
              checkpoint_control: Optional[CheckpointingControl] = None,
              summary_files: int = 0,
              visualisation: bool = False,
              visualisation_args: Optional[Dict[str, Any]] = None,
              force_terminations_after: int = -1,
              end_timeout: Optional[int] = None,
              split_printstreams: bool = True):
        """
        Generates the environment for a new globally managed parallel optimization job.

        Parameters
        ----------
        task: Callable[[Sequence[float]], float]
            Function to be minimized. Accepts a 1D sequence of parameter values and returns a single value.
            Note: Must be a standalone function which makes no modifications outside of itself.

        bounds: Sequence[Tuple[float, float]]
            Sequence of tuples of the form (min, max) limiting the range of each parameter. Do not use bounds to fix
            a parameter value as this will raise an error. Rather supply fixed parameter values through task_args or
            task_kwargs.

        opt_selector: BaseSelector
            Selection criteria for new optimizers, must be an instance of a BaseSelector subclass. BaseSelector
            subclasses are initialised by default with a set of BaseOptimizer subclasses the user would like to make
            available to the optimization. See BaseSelector and BaseOptimizer documentation for more details.

        working_dir: Union[Path, str] = "."
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

        share_best_solutions: bool = False
            If True the manager will send the best ever seen solution to all its children whenever this is updated.

        hunt_frequency: int = 100
            The number of function calls between successive attempts to evaluate optimizer performance and determine
            if they should be terminated.

        status_frequency: int = 600
            Frequency (in seconds) with which status messages are logged. Note that status messages are delivered
            through a logging INFO level message. Logging must be enabled and setup to see these messages. Consult the
            README for more information.

        checkpoint_control: Optional[CheckpointingControl] = None
            If provided, the manager will use checkpointing during the optimization. This saves its state to disk,
            these files can be used by a new GloMPOManager instance to resume. Checkpointing options are provided
            through a CheckpointingControl instance. (see README for more details on checkpointing).
            Note: Checkpointing requires the use of the dill package for serialisation. If you attempt to checkpoint or
                supply checkpoint controls without this package present, a warning will be raised and no checkpointing
                will occur.

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

        if self.is_initialised:
            warnings.warn("Manager already initialised, cannot reinitialise. Aborting", UserWarning)
            self.logger.warning("Manager already initialised, cannot reinitialise. Aborting")
            return

        # Setup logging
        self.logger.info("Initializing Manager ... ")

        # Setup working directory
        if not isinstance(working_dir, (Path, str)):
            warnings.warn(f"Cannot parse working_dir = {working_dir}. str or bytes expected. Using current "
                          f"work directory.", UserWarning)
            working_dir = "."
        self.working_dir = Path(working_dir).resolve()

        # Save and wrap task
        if not callable(task):
            raise TypeError(f"{task} is not callable.")
        self.task = task
        self.logger.debug("Task wrapped successfully")

        # Save optimizer selection criteria
        if isinstance(opt_selector, BaseSelector):
            self.opt_selector = opt_selector
        else:
            raise TypeError("opt_selector not an instance of a subclass of BaseSelector.")

        # Save bounds
        if is_bounds_valid(bounds, raise_invalid=True):
            self.bounds = [Bound(*bnd) for bnd in bounds]
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

        # Save killing conditions
        if killing_conditions:
            if isinstance(killing_conditions, BaseHunter):
                self.killing_conditions = killing_conditions
            else:
                raise TypeError("killing_conditions not an instance of a subclass of BaseHunter.")
        else:
            self.killing_conditions = None
            self.logger.info("Hunting will not be used by the manager.")

        # Save behavioural args
        self.allow_forced_terminations = force_terminations_after > 0
        self._too_long = force_terminations_after
        self.summary_files = np.clip(int(summary_files), 0, 5)
        if self.summary_files != summary_files:
            self.logger.warning(f"summary_files argument given as {summary_files} clipped to {self.summary_files}")
        self.split_printstreams = bool(split_printstreams)
        self.overwrite_existing = bool(overwrite_existing)
        self.visualisation = visualisation
        self.hunt_frequency = hunt_frequency
        self.spawning_opts = True
        self.incumbent_sharing = share_best_solutions
        self.status_frequency = int(status_frequency)

        # Setup Checkpointing
        if isinstance(checkpoint_control, CheckpointingControl):
            if HAS_DILL:
                self.checkpoint_control = checkpoint_control
            else:
                self.logger.warning("Checkpointing controls ignored. Cannot setup infrastructure without dill package.")
                warnings.warn("Checkpointing controls ignored. Cannot setup infrastructure without dill package.",
                              UserWarning)
                self.checkpoint_control = None
        else:
            self.checkpoint_control = None

        # Initialise support classes
        if visualisation:
            from .scope import GloMPOScope  # Only imported if needed to avoid matplotlib compatibility issues
            self.visualisation_args = visualisation_args if visualisation_args else {}
            self.scope = GloMPOScope(**visualisation_args) if visualisation_args else GloMPOScope()

        # Setup backend
        if any([backend == valid_opt for valid_opt in ('processes', 'threads', 'processes_forced')]):
            self.proc_backend = 'processes' in backend
            self.opts_daemonic = backend != 'processes_forced'
        else:
            self.proc_backend = True
            self.opts_daemonic = True
            self.logger.warning(f"Unable to parse backend '{backend}'. 'processes' or 'threads' expected."
                                f"Defaulting to 'processes'.")
            warnings.warn(f"Unable to parse backend '{backend}'. 'processes' or 'threads' expected."
                          f"Defaulting to 'processes'.")

        if end_timeout:
            self.end_timeout = end_timeout
        else:
            if self.proc_backend:
                self.end_timeout = 10
            else:
                self.end_timeout = None

        self._is_restart = False

        if self.checkpoint_control and self.checkpoint_control.checkpoint_at_init:
            self.checkpoint()

        self.logger.info("Initialization Done")

    def load_checkpoint(self, path: Union[Path, str],
                        task_loader: Optional[Callable[[Union[Path, str]], Callable[[Sequence[float]], float]]] = None,
                        task: Optional[Callable[[Sequence[float]], float]] = None, **glompo_kwargs):
        """ Initialise GloMPO from the provided checkpoint file and allows an optimization to resume from that point.

            Parameters
            ----------
            path: Union[Path, str]
                Path to GloMPO checkpoint file.

            task_loader: Optional[Callable[[Union[Path, str]], Callable[[Sequence[float]], float]]] = None
                It is possible for checkpoint files to not contain the optimization task within them. This is because
                some functions may be too complex to reduce to a persistent state and need to be reconstructed. A task
                loader method can be executed to construct a task from the files dumped in the checkpoint by
                task.checkpoint_save when the checkpoint was constructed. (see documentation for GloMPOManger.checkpoint
                method for more on this).

                task_loader accepts a path to the files task.checkpoint_save constructed and returns a callable which is
                the task itself.

                If both task_loader and task are provided, the manager will first attempt to use the task_loader and
                then only use task if that fails otherwise task is ignored.

            task: Optional[Callable[[Sequence[float]], float]] = None
                In the case that the checkpoint does not contain a record of the task, it can be provided directly here.
                If both task_loader and task are provided, the manager will first attempt to use the task_loader and
                then only use task if that fails otherwise task is ignored.

            glompo_kwargs
                Most arguments supplied to GloMPOManager.setup can also be provided here. This will overwrite the values
                saved in the checkpoint.
                Arguments which cannot/should not be changed:
                    bounds: Many optimizers save the bounds during checkpointing. If changed here old optimizers will
                        retain the old bounds but new optimizers will start in new bounds.
                    max_jobs: If this is decreased and falls below the number required by the optimizers in the
                        checkpoint, the manager will attempt to adjust the workers for each optimizer to fit the new
                        limit. Slots are apportioned equally (regardless of the distribution in the checkpoint) and
                        there is no guarantee that the optimizers will actually respond to this change.
                    visualisation_args: Due to the semantics of constructions these arguments will not be accepted by
                        the loaded scope object.

            Notes
            -----
            GloMPO produces the requested log files when it closes (ie a convergence or crash). The working directory
            is, however, purged of old results at the start of the optimization (if overwriting is allowed). This
            behavior is the same regardless of whether the optimization is a resume or a fresh start. This means it is
            the users responsibility to save and move important files from the working_dir before a resume. This is
            particularly important for optimizer printstreams (which are overwritten) as well as movie files which can
            later be stitched together to make a single video of the entire optimization.

            GloMPO does not support making a single continuous recording of the optimization if it is stopped and
            resumed at some point. However, at the end of each section a movie file is made and these can be stitched
            together to make a continuous recording.
        """

        if self.is_initialised:
            warnings.warn("Manager already initialised, cannot reinitialise. Aborting", UserWarning)
            self.logger.warning("Manager already initialised, cannot reinitialise. Aborting")
            return

        path = Path(path).resolve()
        self.logger.info(f"Initializing from Checkpoint: {path}")

        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = Path(tmp_dir_obj.name)

        with tarfile.open(path, 'r:gz') as tfile:
            tfile.extractall(tmp_dir)

        # Load manager variables
        try:
            with (tmp_dir / 'manager').open('rb') as file:
                data = dill.load(file)
                for var, val in data.items():
                    try:
                        setattr(self, var, val)
                    except Exception as e:
                        raise CheckpointingError(f"Could not set {var} attribute correctly") from e
        except Exception as e:
            raise CheckpointingError("Error loading manager. Aborting.") from e

        # Setup Task
        try:
            self.task = None
            if (tmp_dir / 'task').exists():
                with (tmp_dir / 'task').open('rb') as file:
                    try:
                        self.task = dill.load(file)
                        self.logger.info("Task successfully unpickled")
                    except PickleError as e:
                        self.logger.error("Unpickling task failed.")
                        raise e
            else:
                self.logger.warning('No task detected in checkpoint, task or task_loader required.')

            if not self.task and task_loader:
                try:
                    self.task = task_loader(tmp_dir)
                    assert callable(self.task)
                    self.logger.info("Task successfully loaded.")
                except Exception as e:
                    self.logger.error("Use of task_loader failed.")
                    raise e

            if not self.task and task:
                try:
                    self.task = task
                    assert callable(self.task)
                except AssertionError as e:
                    self.logger.error("Could not set task, not callable")
                    raise e

            assert self.task is not None

        except Exception as e:
            raise CheckpointingError("Failed to build task due to error") from e

        # Allow manual overrides
        permit_keys = dir(self)
        for key, val in glompo_kwargs.items():
            if key == 'backend':
                backend = glompo_kwargs['backend']
                self.proc_backend = 'processes' in backend
                self.opts_daemonic = backend != 'processes_forced'
            elif key == 'force_terminations_after':
                force_terminations_after = glompo_kwargs['force_terminations_after']
                self.allow_forced_terminations = force_terminations_after > 0
                self._too_long = force_terminations_after
            elif key == 'visualisation_args':
                pass
            elif key == 'working_dir':
                self.working_dir = Path(val).resolve()
            elif key in permit_keys:
                setattr(self, key, val)
            else:
                self.logger.warning(f"Cannot parse keyword argument '{key}'. Ignoring.")

        # Extract scope and rebuild writer if still visualizing
        if self.visualisation:
            from .scope import GloMPOScope

            if (tmp_dir / 'scope').exists():
                self.logger.info('Scope checkpoint found, extracting')
                self.scope = GloMPOScope()
                try:
                    self.scope.load_state(tmp_dir)
                except Exception as e:
                    warnings.warn("Could not load scope, building fresh.", RuntimeWarning)
                    self.scope = GloMPOScope(**self.visualisation_args)
            else:
                self.scope = GloMPOScope(**self.visualisation_args)

        # Modify/create missing variables
        assert len(self.dt_starts) == len(self.dt_ends), "Timestamps missing from checkpoint."
        self._optimizer_packs: Dict[int, ProcessPackage] = {}
        self.t_used = sum([(end - start).total_seconds() for start, end in zip(self.dt_starts, self.dt_ends)])
        self.t_start = None
        self.t_end = None
        self.opt_crashed = False
        self.last_opt_spawn = (0, 0)
        # noinspection PyBroadException
        try:
            self.converged = self.convergence_checker(self)
        except Exception:
            self.converged = False
        if self.converged:
            self.logger.warning(f"The convergence criteria already evaluates to True. The manager will be unable to"
                                f" resume the optimization. Consider changing the convergence criteria.\n"
                                f"{nested_string_formatting(self.convergence_checker.str_with_result())}")
            warnings.warn("The convergence criteria already evaluates to True. The manager will be unable to resume"
                          " the optimization. Consider changing the convergence criteria.", RuntimeWarning)

        # Append nan to histories to show break in optimizations
        self.cpu_history.append(float('nan'))
        self.mem_history.append(float('nan'))
        self.load_history.append((float('nan'),) * 3)

        # Load optimizer state
        restarts = {int(opt.name): self._opt_checkpoints[int(opt.name)].slots for opt in
                    (tmp_dir / 'optimizers').iterdir()}
        if self.max_jobs < sum(restarts.values()):
            self.logger.warning("The maximum number of jobs allowed is less than that demanded by the optimizers in "
                                "the checkpoint. Attempting to adjust the number of workers in each optimizer to fit. "
                                "Jobs are divided equally and there is no guarantee the optimizers will respond as "
                                "expected.")
            warnings.warn("The maximum number of jobs allowed is less than that demanded by the optimizers in "
                          "the checkpoint. Attempting to adjust the number of workers in each optimizer to fit. "
                          "Jobs are divided equally and there is no guarantee the optimizers will respond as "
                          "expected.", UserWarning)
            new_slots = int(self.max_jobs / len(restarts))
            if new_slots < 1:
                raise CheckpointingError("Insufficient max_jobs allowed to restart all optimizers in checkpoint.")
            restarts = {opt_id: new_slots for opt_id in restarts}

        # Rebuild child processes
        backend = 'threads' if self.opts_daemonic else 'processes'
        for opt_id, slots in restarts.items():
            parent_pipe, child_pipe = mp.Pipe()
            event = self._mp_manager.Event()
            event.set()
            try:
                opt_class = self._opt_checkpoints[opt_id].opt_type
                optimizer = opt_class.checkpoint_load(opt_id=opt_id,
                                                      signal_pipe=child_pipe,
                                                      results_queue=self.optimizer_queue,
                                                      pause_flag=event,
                                                      workers=slots,
                                                      backend=backend,
                                                      path=tmp_dir / 'optimizers' / f'{opt_id:04}')

                optimizer.workers = slots
                optimizer._backend = backend  # Overwrite in case load_state set old values

                x0 = [0] * self.n_parms  # Ignored during restart
                bounds = np.array(self.bounds)  # Ignored during restart
                # noinspection PyProtectedMember
                target = optimizer._minimize

                if self.split_printstreams and self.proc_backend:
                    # noinspection PyProtectedMember
                    target = process_print_redirect(opt_id, self.working_dir, optimizer._minimize)

                kwargs = {'target': target,
                          'args': (self.task, x0, bounds),
                          'name': f"Opt{opt_id}",
                          'daemon': self.opts_daemonic}
                if self.proc_backend:
                    process = mp.Process(**kwargs)
                else:
                    process = CustomThread(working_directory=self.working_dir,
                                           redirect_print=self.split_printstreams, **kwargs)

                self._optimizer_packs[opt_id] = ProcessPackage(process, parent_pipe, event, slots)

                if self.visualisation and opt_id not in self.scope.opt_streams:
                    self.scope.add_stream(opt_id, type(optimizer).__name__)

            except Exception as e:
                self.logger.error(f"Failed to initialise optimizer {opt_id}", exc_info=e)
                warnings.warn(f"Failed to initialise optimizer {opt_id}: {e}", RuntimeWarning)

        if len(self._optimizer_packs) == 0 and len(restarts) > 0:
            raise CheckpointingError("Unable to successfully built any optimizers from the checkpoint.")

        self._is_restart = True
        tmp_dir_obj.cleanup()

        self.logger.info("Initialization Done")

    def start_manager(self) -> Result:
        """ Begins the optimization routine and returns the selected minimum in an instance of MinimizeResult. """

        if not self.is_initialised:
            self.logger.error("Cannot start manager, initialise manager first with setup or load_checkpoint")
            warnings.warn("Cannot start manager, initialise manager first with setup or load_checkpoint", UserWarning)
            return Result([], float('inf'), {}, {})

        caught_exception = None

        # Check convergence criteria
        # noinspection PyBroadException
        try:
            # Attempt to evaluate the convergence checker, may fail since the manager has not started yet
            checker_condition = self.convergence_checker(self)
            reason = self.convergence_checker.str_with_result() if checker_condition else \
                "No optimizers alive, spawning stopped."
            self.converged = checker_condition or (len(self._optimizer_packs) == 0 and not self.spawning_opts)
            if self.converged:
                self.logger.warning("Convergence conditions met before optimizer start. Aborting start.")
                warnings.warn("Convergence conditions met before optimizer start. Aborting start.", RuntimeWarning)
                return self.result
        except Exception:
            reason = "None"
            self.converged = False

        # Make working dir
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self._purge_old_results()

        if self.visualisation and self.scope.record_movie:
            self.scope.setup_moviemaker(self.working_dir)

        if self.split_printstreams:
            (self.working_dir / "glompo_optimizer_printstreams").mkdir(exist_ok=True)
            if not self.proc_backend:
                sys.stdout = ThreadPrintRedirect(sys.stdout)
                sys.stderr = ThreadPrintRedirect(sys.stderr)

        # Setup system monitoring
        if HAS_PSUTIL:
            self._setup_system_monitoring()

        # Settings check
        if self.allow_forced_terminations and not self.proc_backend:
            warnings.warn("Cannot use force terminations with threading.", UserWarning)
            self.logger.warning("Cannot use force terminations with threading.")

        try:
            self.logger.info("Starting GloMPO Optimization Routine")

            self.t_start = time()
            self.last_status = self.t_start
            self.last_time_checkpoint = self.t_start
            self.dt_starts.append(datetime.fromtimestamp(self.t_start))

            # Restart specific tasks
            if self._is_restart:
                for opt_id, pack in self._optimizer_packs.items():
                    pack.process.start()
                    self._last_feedback[opt_id] = time()

            while not self.converged:
                self.logger.debug("Checking for available optimizer slots")
                self._fill_optimizer_slots()
                self.logger.debug("New optimizer check done")

                self.logger.debug("Checking optimizer signals")
                for opt_id in self._optimizer_packs:
                    self._check_signals(opt_id)
                self.logger.debug("Signal check done.")

                self.logger.debug("Checking optimizer iteration results")
                self._process_results(10)
                self.logger.debug("Iteration results check done.")

                self.logger.debug("Checking for user interventions.")
                self._is_manual_shutdowns()
                self._is_manual_checkpoints()

                self.logger.debug("Checking for hanging processes")
                self._inspect_children()

                # Purge old processes
                for opt_id, pack in [*self._optimizer_packs.items()]:
                    if not pack.process.is_alive() and opt_id in self._graveyard:
                        del self._optimizer_packs[opt_id]

                all_dead = len([p for p in self._optimizer_packs.values() if p.process.is_alive()]) == 0
                checker_condition = self.convergence_checker(self)

                if checker_condition:
                    reason = self.convergence_checker.str_with_result()

                self.converged = checker_condition or (all_dead and not self.spawning_opts)
                if self.converged:
                    self.logger.info("Convergence Reached")

                if time() - self.last_status > self.status_frequency:
                    self.logger.info(self._build_status_message())

                if self.checkpoint_control:
                    if time() - self.last_time_checkpoint > self.checkpoint_control.checkpoint_time_frequency:
                        self.last_time_checkpoint = time()
                        self.checkpoint()
                    elif self.f_counter - self.last_iter_checkpoint > self.checkpoint_control.checkpoint_iter_frequency:
                        self.last_iter_checkpoint = self.f_counter
                        self.checkpoint()

            self.logger.info("Exiting manager loop")
            self.logger.info(f"Exit conditions met: \n"
                             f"{nested_string_formatting(reason)}")

            if self.checkpoint_control and self.checkpoint_control.checkpoint_at_conv:
                self.checkpoint()

            self.logger.debug("Cleaning up multiprocessing")
            self._stop_all_children()

        except KeyboardInterrupt:
            caught_exception = "User Interrupt"
            reason = caught_exception
            self.logger.error("Caught User Interrupt, closing GloMPO gracefully.")
            warnings.warn("Optimization failed. Caught User Interrupt", RuntimeWarning)
            self._stop_all_children("User Interrupt")

        except Exception as e:
            caught_exception = "".join(traceback.TracebackException.from_exception(e).format())
            reason = "GloMPO Crash"
            self.logger.critical("Critical error encountered. Attempting to close GloMPO gracefully", exc_info=e)
            warnings.warn(f"Optimization failed. Caught exception: {e}", RuntimeWarning)
            self._stop_all_children("GloMPO Crash")

        finally:

            self.logger.info("Cleaning up and closing GloMPO")

            self.t_end = time()
            dt_end = datetime.fromtimestamp(self.t_end)
            if len(self.dt_starts) == len(self.dt_ends):
                self.dt_ends[-1] = dt_end
            else:
                self.dt_ends.append(dt_end)

            if self.visualisation:
                if self.scope.record_movie and not caught_exception:
                    self.logger.debug("Generating movie")
                    self.scope.generate_movie()
                self.scope.close_fig()

            self.logger.debug("Saving summary file results")
            self._save_log(self.result, reason, caught_exception, self.working_dir, self.summary_files)

            self.result = Result(self.result.x,
                                 self.result.fx,
                                 {**self.result.stats, 'end_cond': reason} if self.result.stats else {
                                     'end_cond': reason},
                                 self.result.origin)

            if not self.proc_backend and self.split_printstreams:
                sys.stdout.close()
                sys.stderr.close()

            self.logger.info("GloMPO Optimization Routine Done")

            return self.result

    def checkpoint(self):
        """ Saves the state of the manager and any existing optimizers to disk. GloMPO can be loaded from these files
            and resume optimization from this state.

            Notes
            -----
            When checkpointing GloMPO will attempt to handle the task in three ways:
                1) Pickle the with the other manager variables, this is the easiest and most straightforward method.

                2) If the above fails, the manager will attempt to call task.checkpoint_save(path) if it is present.
                   This is expected to create file/s in path which is suitable for reconstruction during
                   load_checkpoint(). When resuming a run the manager will attempt to reconstruct the task by calling
                   the method passed to the task_loader method of load_checkpoint.

                3) If the manager cannot perform either of the above methods the checkpoint will be constructed without
                   a task. In that case a fully initialised task must be given to load_checkpoint.
        """

        self.logger.info("Constructing Checkpoint")

        # Construct Checkpoint Name
        path = self.checkpoint_control.checkpointing_dir / self.checkpoint_control.get_name()
        path.mkdir(parents=True, exist_ok=True)

        overwriting_chkpt = False
        ovw_path = path.parent / '_overwriting_chkpt.tar.gz'

        try:
            # Save timestamp and checkpoint name
            if len(self.dt_starts) > 0:
                if len(self.dt_starts) == len(self.dt_ends):
                    self.dt_ends[-1] = datetime.now()
                else:
                    self.dt_ends.append(datetime.now())
            self.checkpoint_history.add(str(path.resolve().with_suffix('.tar.gz')))

            self._checkpoint_optimizers(path)
            self._checkpoint_manager(path)
            self._checkpoint_task(path)

            # Save scope
            if self.visualisation and self.scope:
                self.scope.checkpoint_save(path)
            self.logger.debug("Scope successfully pickled")

            # Compress checkpoint
            self.logger.debug("Building TarFile")
            tar_path = path.with_suffix('.tar.gz')
            if tar_path.exists():
                self.logger.warning("Overwriting existing checkpoint. To avoid this change the checkpointing naming "
                                    "format")
                warnings.warn("Overwriting existing checkpoint. To avoid this change the checkpointing naming "
                              "format")
                tar_path.replace(ovw_path)
                overwriting_chkpt = True

            try:
                with tarfile.open(tar_path, 'x:gz') as tfile:
                    tfile.add(path, recursive=True, arcname='')
                self.logger.debug("TarFile built")
            except tarfile.TarError as e:
                self.logger.error("Error encountered during compression.")
                if overwriting_chkpt:
                    self.logger.info("Overwritten checkpoint restored")
                    ovw_path.replace(tar_path)
                raise CheckpointingError("Could not compress checkpoint", e)

            # Delete old checkpoints
            if self.checkpoint_control.keep_past > -1:
                self.logger.debug("Finding old checkpoints to delete")
                files = (file.name for file in self.checkpoint_control.checkpointing_dir.iterdir())
                to_delete = sorted(filter(self.checkpoint_control.matches_naming_format, files), reverse=True)
                self.logger.debug(f"Identified to delete: {to_delete[self.checkpoint_control.keep_past + 2:]}")
                for old in to_delete[self.checkpoint_control.keep_past + 2:]:
                    del_path = self.checkpoint_control.checkpointing_dir / old
                    if del_path.is_file():
                        del_path.unlink()

        except CheckpointingError as e:
            caught_exception = "".join(traceback.TracebackException.from_exception(e).format())
            self.checkpoint_history.remove(str(path.resolve().with_suffix('.tar.gz')))

            if self.checkpoint_control.raise_checkpoint_fail:
                self.logger.error(f"Checkpointing failed: {caught_exception}")
                raise e

            self.logger.warning(f"Checkpointing failed: {caught_exception}.\nAborting checkpoint construction.")
            warnings.warn(f"Checkpointing failed: {caught_exception}.\nAborting checkpoint construction.")
        finally:
            shutil.rmtree(path, ignore_errors=True)
            if ovw_path.exists():
                ovw_path.unlink()

        if self.converged:
            [pack.signal_pipe.send(1) for _, pack in self._optimizer_packs.items() if pack.process.is_alive()]
        self._toggle_optimizers(1)
        self.logger.info(f"Checkpoint '{path.name}' successfully built")

    def dump_state(self, summary_files: Optional[int] = None, dump_dir: Optional[Path] = None):
        """ Produces the same output files produced by start_manager without actually running any optimization. Useful
            to extract output files from checkpoints.

            Parameters
            ----------
            summary_files: Optional[int] = None
                If provided, this will overwrite the manager summary_files setting allowing this method to produce more
                or less output as desired. If not provided and the manager summary_files setting is not set then this
                method will pass and produce no output. This would typically only happen if trying to dump an
                uninitialised manager.
            dump_dir: Optional[Path] = None
                If provided, this will overwrite the manager working_dir allowing the output to be redirected to a
                different folder so as to not interfere with files in the working directory.
        """

        self.logger.info("Dumping manager state")
        if dump_dir:
            dump_dir = Path(dump_dir).resolve()
            dump_dir.mkdir(exist_ok=True)
        else:
            dump_dir = self.working_dir

        if summary_files is None:
            summary_files = self.summary_files
        if summary_files is None:  # Possible if self.summary_files has not been set
            summary_files = 0

        self._save_log(self.result, "Manual Save State", None, dump_dir, summary_files)

    """ Management Sub-Tasks """

    def _fill_optimizer_slots(self):
        """ Starts new optimizers if there are slots available. """

        processes = [pack.slots for pack in self._optimizer_packs.values() if pack.process.is_alive()]
        count = sum(processes)

        if self.last_opt_spawn[0] == self.f_counter and \
                self.o_counter > self.last_opt_spawn[1] + 5 and \
                self.opt_crashed:
            raise RuntimeError("Optimizers spawning and crashing immediately.")

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

        processes = [pack.slots for pack in self._optimizer_packs.values() if pack.process.is_alive()]
        if started_new:
            self.last_opt_spawn = (self.f_counter, self.o_counter) \
                if self.last_opt_spawn[0] != self.f_counter else self.last_opt_spawn
            f_best = f'{self.result.fx:.3E}' if self.result.fx is not None else None
            self.logger.info(f"Status: {len(processes)} optimizers alive, {sum(processes)}/{self.max_jobs} slots filled"
                             f", {self.f_counter} function evaluations, f_best = {f_best}.")
        elif len(processes) == 0:
            raise RuntimeError("Not enough worker slots to start any optimizers with the current settings.")

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

        if self.split_printstreams and self.proc_backend:
            # noinspection PyProtectedMember
            target = process_print_redirect(opt_id, self.working_dir, optimizer._minimize)

        kwargs = {'target': target,
                  'args': (task, x0, bounds),
                  'kwargs': call_kwargs,
                  'name': f"Opt{opt_id}",
                  'daemon': self.opts_daemonic}
        if self.proc_backend:
            process = mp.Process(**kwargs)
        else:
            process = CustomThread(working_directory=self.working_dir, redirect_print=self.split_printstreams, **kwargs)

        self._optimizer_packs[opt_id] = ProcessPackage(process, pipe, event, workers)
        self._optimizer_packs[opt_id].process.start()
        self._last_feedback[opt_id] = time()

        if self.visualisation and opt_id not in self.scope.opt_streams:
            self.scope.add_stream(opt_id, type(optimizer).__name__)

    def _setup_new_optimizer(self, slots_available: int) -> Optional[OptimizerPackage]:
        """ Selects and initializes new optimizer and multiprocessing variables. Returns an OptimizerPackage which
            can be sent to _start_new_job to begin new process. Returns None if an optimizer satisfying the number of
            available slots or spawning conditions is not found.
        """

        selector_return = self.opt_selector.select_optimizer(self, self.opt_log, slots_available)

        if not selector_return:
            if selector_return is False:
                self.logger.info("Optimizer spawning deactivated.")
                self.spawning_opts = False
            return None

        selected, init_kwargs, call_kwargs = selector_return
        if not self.proc_backend:
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

        self.opt_log.add_optimizer(self.o_counter, type(optimizer).__name__, str(datetime.now()))
        self._opt_checkpoints[self.o_counter] = OptimizerCheckpoint(selected, init_kwargs['workers'])

        if call_kwargs:
            return OptimizerPackage(self.o_counter, optimizer, call_kwargs, parent_pipe, event, init_kwargs['workers'])
        return OptimizerPackage(self.o_counter, optimizer, {}, parent_pipe, event, init_kwargs['workers'])

    def _check_signals(self, opt_id: int) -> bool:
        """ Checks for signals from optimizer opt_id and processes it.
            Returns a bool indicating whether a signal was found.
        """

        pipe = self._optimizer_packs[opt_id].signal_pipe
        found_signal = False
        if opt_id not in self._graveyard and pipe.poll():
            try:
                key, message = pipe.recv()
                self._last_feedback[opt_id] = time()
                self.logger.info(f"Signal {key} from {opt_id}.")
                if key == 0:
                    self.opt_log.put_metadata(opt_id, "Stop Time", str(datetime.now()))
                    self.opt_log.put_metadata(opt_id, "End Condition", message)
                    self._graveyard.add(opt_id)
                    self.conv_counter += 1
                elif key == 9:
                    self.opt_log.put_message(opt_id, message)
                    self.logger.warning(f"Optimizer {opt_id} Exception: {message}")
                    self.opt_crashed = "Traceback" in message or self.opt_crashed
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

        for opt_id, pack in self._optimizer_packs.items():

            # Find dead optimizer processes that did not properly signal their termination.
            if opt_id not in self._graveyard and not pack.process.is_alive():
                exitcode = pack.process.exitcode
                if exitcode == 0:
                    if not self._check_signals(opt_id):
                        self.conv_counter += 1
                        self._graveyard.add(opt_id)
                        self.opt_log.put_message(opt_id, "Terminated normally without sending a minimization "
                                                         "complete signal to the manager.")
                        warnings.warn(f"Optimizer {opt_id} terminated normally without sending a "
                                      f"minimization complete signal to the manager.", RuntimeWarning)
                        self.logger.warning(f"Optimizer {opt_id} terminated normally without sending a "
                                            f"minimization complete signal to the manager.")
                        self.opt_log.put_metadata(opt_id, "Approximate Stop Time", str(datetime.now()))
                        self.opt_log.put_metadata(opt_id, "End Condition", "Normal termination (Reason unknown)")
                else:
                    self._graveyard.add(opt_id)
                    self.opt_log.put_message(opt_id, f"Terminated in error with code {-exitcode}")
                    warnings.warn(f"Optimizer {opt_id} terminated in error with code {-exitcode}",
                                  RuntimeWarning)
                    self.logger.error(f"Optimizer {opt_id} terminated in error with code {-exitcode}")
                    self.opt_log.put_metadata(opt_id, "Approximate Stop Time", str(datetime.now()))
                    self.opt_log.put_metadata(opt_id, "End Condition", f"Error termination (exitcode {-exitcode}).")

            # Find hanging processes
            if pack.process.is_alive() and \
                    time() - self._last_feedback[opt_id] > self._too_long and \
                    self.allow_forced_terminations and \
                    opt_id not in self.hunt_victims and \
                    self.proc_backend:
                warnings.warn(f"Optimizer {opt_id} seems to be hanging. Forcing termination.", RuntimeWarning)
                self.logger.error(f"Optimizer {opt_id} seems to be hanging. Forcing termination.")
                self._graveyard.add(opt_id)
                self.opt_log.put_message(opt_id, "Force terminated due to no feedback timeout.")
                self.opt_log.put_metadata(opt_id, "Approximate Stop Time", str(datetime.now()))
                self.opt_log.put_metadata(opt_id, "End Condition", "Forced GloMPO Termination")
                pack.process.terminate()

            # Force kill zombies
            if opt_id in self.hunt_victims and \
                    self.allow_forced_terminations and \
                    pack.process.is_alive() and \
                    time() - self.hunt_victims[opt_id] > self._too_long and \
                    self.proc_backend:
                pack.process.terminate()
                pack.process.join(3)
                self.opt_log.put_message(opt_id, "Force terminated due to no feedback after kill signal "
                                                 "timeout.")
                self.opt_log.put_metadata(opt_id, "Approximate Stop Time", str(datetime.now()))
                self.opt_log.put_metadata(opt_id, "End Condition", "Forced GloMPO Termination")
                warnings.warn(f"Forced termination signal sent to optimizer {opt_id}.", RuntimeWarning)
                self.logger.error(f"Forced termination signal sent to optimizer {opt_id}.")

    def _process_results(self, max_results: Optional[int] = None) -> Tuple[List[IterationResult], Set[int]]:
        """ Retrieve results from the queue and process them into the opt_log.
            If max_results is provided, accept at most this number of results.
            Otherwise will loop until the queue is empty.

            Returns a tuple of:
                - a list of accepted results
                - a list of hunt victims.
        """

        results_accepted = []
        victims = set()
        if max_results:
            def condition():
                return len(results_accepted) < max_results
        else:
            def condition():
                return not self.optimizer_queue.empty()

        while condition():
            try:
                res = self.optimizer_queue.get(block=True, timeout=1)
                results_accepted.append(res)
            except queue.Empty:
                self.logger.debug("Timeout on result queue.")
                break

            self._last_feedback[res.opt_id] = time()
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

                # Start hunt if required
                best_id = -1
                self.result = self._update_best_result()
                if self.result.origin and 'opt_id' in self.result.origin:
                    best_id = self.result.origin['opt_id']

                if best_id > 0 and self.killing_conditions and self.f_counter - self.last_hunt >= self.hunt_frequency:
                    [victims.add(vic) for vic in self._start_hunt(best_id)]

        return results_accepted, victims

    def _start_hunt(self, hunter_id: int) -> Set[int]:
        """ Creates a new hunt with the provided hunter_id as the 'best' optimizer looking to terminate
            the other active optimizers according to the provided killing_conditions.
        """

        self.hunt_counter += 1
        self.last_hunt = self.f_counter
        victims = set()  # IDs of hunt victims

        self.logger.debug("Starting hunt")
        for victim_id in self._optimizer_packs:
            in_graveyard = victim_id in self._graveyard
            has_points = len(self.opt_log.get_history(victim_id, "fx")) > 0
            if not in_graveyard and has_points and victim_id != hunter_id:
                self.logger.debug(f"Optimizer {hunter_id} -> Optimizer {victim_id} hunt started.")
                kill = self.killing_conditions(self.opt_log, hunter_id, victim_id)

                if kill:
                    reason = nested_string_formatting(self.killing_conditions.str_with_result())
                    self.logger.info(f"Optimizer {hunter_id} wants to kill Optimizer {victim_id}:\n"
                                     f"Reason:\n"
                                     f"{reason}")

                    if victim_id not in self._graveyard:
                        self._shutdown_job(victim_id, hunter_id, reason)
                        victims.add(victim_id)

        self.logger.debug("Hunting complete")
        return victims

    def _is_manual_shutdowns(self):
        """ If a file titled STOP_x is found in the working directory then the manager will shutdown optimizer x. """

        stop_files = (x for x in self.working_dir.iterdir() if re.match("STOP_[0-9]+", x.name))
        for file in stop_files:
            file.unlink()
            try:
                _, opt_id = file.name.split('_')
                opt_id = int(opt_id)
                if opt_id not in self._optimizer_packs or opt_id in self._graveyard:
                    self.logger.info(f"Matching living optimizer not found for '{file}'")
                    continue

                self._shutdown_job(opt_id, None, "User STOP file intervention.")
                self.logger.info(f"STOP file found for Optimizer {opt_id}")
            except ValueError as e:
                self.logger.debug(f"Error encountered trying to process STOP file '{file}'.", exc_info=e)
                continue

    def _is_manual_checkpoints(self):
        """ If a file titled CHKPT is found in the working directory then the manager will perform an immediate
            unscheduled checkpoint.
        """

        chkpt_path = self.working_dir / "CHKPT"
        if chkpt_path.exists():
            chkpt_path.unlink()

            has_controls = bool(self.checkpoint_control)
            if not has_controls:
                self.logger.warning("Manual checkpoint requested but checkpointing control not setup during "
                                    "initialisation, constructing with defaults.")
                self.checkpoint_control = CheckpointingControl()

            self.logger.info("Manual checkpoint requested")
            self.checkpoint()

            if not has_controls:
                self.checkpoint_control = None

    def _shutdown_job(self, opt_id: int, hunter_id: Optional[int], reason: str):
        """ Sends a stop signal to optimizer opt_id and updates variables around its termination. """

        self.hunt_victims[opt_id] = time()
        self._graveyard.add(opt_id)

        self._optimizer_packs[opt_id].signal_pipe.send(1)
        self.logger.debug(f"Termination signal sent to {opt_id}")

        self.opt_log.put_metadata(opt_id, "Stop Time", str(datetime.now()))
        self.opt_log.put_metadata(opt_id, "End Condition", LiteralWrapper(f"GloMPO Termination\n"
                                                                          f"Hunter: {hunter_id}\n"
                                                                          f"Reason: \n"
                                                                          f"{reason}"))

        if self.visualisation:
            self.scope.update_kill(opt_id)

    def _update_best_result(self) -> Result:
        """ Returns the best results found in the log of results. """

        best_iter = self.opt_log.best_iter

        if self.incumbent_sharing and (not self.result.fx or best_iter['fx'] < self.result.fx):
            for opt_id, pack in self._optimizer_packs.items():
                if opt_id != best_iter['opt_id'] and pack.process.is_alive():
                    pack.signal_pipe.send((4, best_iter['x'], best_iter['fx']))

        best_origin = {"opt_id": best_iter['opt_id'],
                       "type": self.opt_log.get_metadata(best_iter['opt_id'], "Optimizer Type")}

        best_stats = {'f_evals': self.f_counter,
                      'opts_started': self.o_counter,
                      'opts_killed': len(self.hunt_victims),
                      'opts_conv': self.conv_counter,
                      'end_cond': None}

        return Result(best_iter['x'], best_iter['fx'], best_stats, best_origin)

    def _stop_all_children(self, crash_reason: Optional[str] = None):
        """ Shuts down and cleansup all active children """

        # Attempt to send shutdown signals
        try:
            message = (1, "GloMPO Crash") if crash_reason else 1
            [pack.signal_pipe.send(message) for _, pack in self._optimizer_packs.items() if pack.process.is_alive()]
        except Exception as e:
            self.logger.debug("Pipe messaging failed during cleanup.", exc_info=e)

        # Purge the queue to ensure no optimizers are blocking
        try:
            while not self.optimizer_queue.empty() and not crash_reason:
                self.optimizer_queue.get_nowait()
        except Exception as e:
            # Queue may not be accessible in a crash
            self.logger.debug("Queue purge failed.", exc_info=e)

        for opt_id, pack in self._optimizer_packs.items():
            # Add stop condition to logs without overwriting existing ones
            try:
                self.opt_log.get_metadata(opt_id, "End Condition")
                self.opt_log.get_metadata(opt_id, "Stop Time")
            except KeyError:
                self._graveyard.add(opt_id)
                self.opt_log.put_metadata(opt_id, "Stop Time", str(datetime.now()))
                self.opt_log.put_metadata(opt_id, "End Condition",
                                          crash_reason if crash_reason else "GloMPO Convergence")

            if pack.process.is_alive():
                pack.process.join(timeout=self.end_timeout if not crash_reason else 0.1)
                if pack.process.is_alive():
                    if self.proc_backend:
                        self.logger.info(f"Termination signal sent to optimizer {opt_id}")
                        pack.process.terminate()
                    else:
                        self.logger.warning(f"Could not join optimizer {opt_id}. May crash out with it still running "
                                            f"and thus generate errors. Terminations cannot be sent to threads.")

    def _save_log(self, result: Result, reason: str, caught_exception: Optional[str], dump_dir: Path,
                  summary_files: int):
        """ Saves the manager's state and history into the collection of files requested by summary_files. """

        if summary_files > 0:
            if caught_exception:
                reason = f"Process Crash: {caught_exception}"

            if HAS_PSUTIL and self._process:
                cores = self._process.cpu_affinity()
                resource_summary = self._summarise_resource_usage()

                run_info = {
                    "Memory": {
                        "Used": {
                            "Max": resource_summary['mem_max'],
                            "Ave": resource_summary['mem_ave']},
                        "Available": present_memory(psutil.virtual_memory().total)},
                    "CPU": {
                        "Cores": {
                            "Total": len(cores),
                            "IDs": FlowList(cores)},
                        "Frequency":
                            f"{psutil.cpu_freq().max / 1000}GHz",
                        "Load": {
                            "Average": FlowList(resource_summary['load_ave']),
                            "Std. Dev.": FlowList(resource_summary['load_std'])},
                        "CPU Usage(%)": {
                            "Average": resource_summary['cpu_ave'],
                            "Std. Dev.": resource_summary['cpu_std']}}}
            else:
                run_info = None

            t_total = str(
                timedelta(seconds=sum([(t - t0).total_seconds() for t0, t in zip(self.dt_starts, self.dt_ends)])))
            t_session = str(timedelta(seconds=self.t_end - self.t_start)) if self.t_start else None
            t_periods = [{"Start": str(t0), "End": str(t)} for t0, t in zip(self.dt_starts, self.dt_ends)]
            data = {
                "Assignment": {
                    "Task": type(self.task).__name__ if isinstance(type(self.task), object) else self.task.__name__,
                    "Working Dir": str(Path.cwd()),
                    "Username": getpass.getuser(),
                    "Hostname": socket.gethostname(),
                    "Time": {"Optimization Periods": t_periods,
                             "Total": t_total,
                             "Session": t_session}},
                "Settings": {"x0 Generator": self.x0_generator,
                             "Convergence Checker": LiteralWrapper(nested_string_formatting(str(
                                 self.convergence_checker))),
                             "Hunt Conditions": LiteralWrapper(nested_string_formatting(str(
                                 self.killing_conditions))) if self.killing_conditions else
                             self.killing_conditions,
                             "Optimizer Selector": self.opt_selector,
                             "Max Jobs": self.max_jobs,
                             "Bounds": BoundGroup(self.bounds)},
                "Counters": {"Function Evaluations": self.f_counter,
                             "Hunts Started": self.hunt_counter,
                             "Optimizers": {"Started": self.o_counter,
                                            "Killed": len(self.hunt_victims),
                                            "Converged": self.conv_counter}}}

            if run_info:
                data["Run Information"] = run_info

            if self.checkpoint_control:
                data["Checkpointing"] = {"Directory": str(self.checkpoint_control.checkpointing_dir.resolve()),
                                         "Checkpoints": list(self.checkpoint_history)}

            data["Solution"] = {"fx": result.fx,
                                "origin": result.origin,
                                "exit cond.": LiteralWrapper(nested_string_formatting(reason)),
                                "x": FlowList(result.x) if result.x else result.x}

            with (dump_dir / "glompo_manager_log.yml").open("w") as file:
                self.logger.debug("Saving manager summary file.")
                yaml.dump(data, file, Dumper=Dumper, default_flow_style=False, sort_keys=False)

            if summary_files >= 2:
                self.logger.debug("Saving optimizers summary file.")
                self.opt_log.save_summary(dump_dir / "opt_best_summary.yml")
            if summary_files >= 3:
                self.logger.debug("Saving optimizer log files.")
                self.opt_log.save_optimizer(dump_dir / "glompo_optimizer_logs")
            if summary_files >= 4:
                self.logger.debug("Saving trajectory plot.")
                signs = set()
                large = 0
                small = float('inf')
                for opt_id in range(1, self.o_counter + 1):
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
                    self.opt_log.plot_trajectory(dump_dir / name, log_scale, best_fx)
            if summary_files == 5:
                self.logger.debug("Saving optimizer parameter trials.")
                self.opt_log.plot_optimizer_trials(dump_dir)

    """ Checkpointing Sub-Tasks """

    def _checkpoint_optimizers(self, path: Union[str, Path]):
        """ Checkpointing sub-task. Gathers, synchronises and saves child optimizers. """

        # Pause optimizers
        messaged = set()
        for opt_id, pack in self._optimizer_packs.items():
            if pack.process.is_alive():
                pack.signal_pipe.send(2)
                messaged.add(opt_id)

        # Synchronise and wait for replies (end or paused)
        not_chkpt = set()  # Set of messaged opts that should not be checkpointed
        wait_reply = messaged.copy()
        living = messaged.copy()
        n_alive = len(messaged)
        while wait_reply:
            self.logger.debug(f"Blocking, {n_alive - len(wait_reply)}/{n_alive} optimizers synced. "
                              f"Waiting on {wait_reply}.")

            if self.optimizer_queue.full():
                accepted, victims = self._process_results(n_alive)  # Free space on queue to avoid blocking
                [not_chkpt.add(res.opt_id) for res in accepted if res.final]
                [not_chkpt.add(vic) for vic in victims]

            for opt_id in wait_reply.copy():
                pack = self._optimizer_packs[opt_id]
                if pack.process.is_alive():
                    if pack.signal_pipe.poll(0.1):
                        key, message = pack.signal_pipe.recv()
                        self.logger.debug(f"Received {key, message} from {opt_id}")
                        if key == 0:
                            self.opt_log.put_metadata(opt_id, "Stop Time", str(datetime.now()))
                            self.opt_log.put_metadata(opt_id, "End Condition", message)
                            self._graveyard.add(opt_id)
                            self.conv_counter += 1
                            not_chkpt.add(opt_id)
                        elif key == 1:
                            if self.visualisation:
                                self.scope.update_pause(opt_id)
                            wait_reply.remove(opt_id)
                        else:
                            raise RuntimeError(f"Unhandled message: {message}")
                else:
                    self.logger.debug(f"Opt {opt_id} dead, removing from wait list")
                    not_chkpt.add(opt_id)
                    wait_reply.remove(opt_id)
                    living.remove(opt_id)
        self.logger.info("Optimizers paused and synced.")

        # Process outstanding results and hunts
        accepted, victims = self._process_results()
        [not_chkpt.add(res.opt_id) for res in accepted if res.final]
        [not_chkpt.add(vic) for vic in victims]
        self.logger.info("Outstanding results processed")

        assert self.optimizer_queue.empty()

        # Send checkpoint_save signals
        (path / 'optimizers').mkdir()
        for opt_id in messaged:
            pack = self._optimizer_packs[opt_id]
            if pack.process.is_alive():
                if opt_id not in not_chkpt:
                    if self.visualisation:
                        self.scope.update_checkpoint(opt_id)
                    pack.signal_pipe.send((0, (path / 'optimizers' / f'{opt_id:04}').absolute()))
                else:
                    pack.signal_pipe.send(3)  # Causes waiting optimizers will pass and not checkpoint

        # Wait for all checkpoint_save to complete
        wait_reply = living.copy()
        while wait_reply:
            for opt_id in wait_reply.copy():
                if not self._optimizer_packs[opt_id].allow_run_event.is_set():
                    wait_reply.remove(opt_id)

        # Confirm all restart files are found
        living_names = {f'{opt_id:04}' for opt_id in messaged - not_chkpt}
        for lv in living_names:
            if not (path / 'optimizers' / lv).exists():
                raise CheckpointingError(f"Unable to identify restart file/folder for optimizer {lv}")
        self.logger.info("All optimizer restart files detected.")

    def _checkpoint_manager(self, path: Union[str, Path]):
        """ Checkpointing sub-task. Pickles essential elements of the manager state. """

        # Select variables for pickling
        pickle_vars = {}
        for var in dir(self):
            val = getattr(self, var)
            if not (callable(getattr(self, var)) and hasattr(val, '__self__')) and \
                    '__' not in var and \
                    not any([var == no_pickle for no_pickle in ('logger', '_process', '_mp_manager',
                                                                '_optimizer_packs', 'scope', 'task',
                                                                'optimizer_queue', 'is_initialised',
                                                                'queue_monitor_arrest', 'queue_monitor_end',
                                                                'queue_monitor_thread')]):
                if dill.pickles(val):
                    pickle_vars[var] = val
                else:
                    raise CheckpointingError(f"Cannot pickle {var}.")

        with (path / 'manager').open('wb') as file:
            try:
                dill.dump(pickle_vars, file)
            except PickleError as e:
                raise CheckpointingError("Could not pickle manager.") from e
        self.logger.debug("Manager successfully pickled")

    def _checkpoint_task(self, path: Union[str, Path]):
        """ Checkpointing sub-task. Identifies procedure to persist minimization task. """

        # Save task
        task_persisted = False

        if not self.checkpoint_control.force_task_save:
            try:
                with (path / 'task').open('wb') as file:
                    dill.dump(self.task, file)
                self.logger.info("Task successfully pickled")
                task_persisted = True
            except PickleError as pckl_err:
                self.logger.info(f"Pickle task failed: {pckl_err}. Attempting task.checkpoint_save()")
                (path / 'task').unlink()

        if not task_persisted:
            try:
                # noinspection PyUnresolvedReferences
                self.task.checkpoint_save(path)
                self.logger.info("Task successfully saved")
            except AttributeError:
                self.logger.info("task.checkpoint_save not found.")
                self.logger.warning("Checkpointing without task.")
            except Exception as e:
                self.logger.warning("Task saving failed", exc_info=e)
                self.logger.warning("Checkpointing without task.")

    """ Sundry Auxiliary Methods """

    def _toggle_optimizers(self, on_off: int):
        """ Sends pause or resume signals to all optimizers based on the on_off parameter:
            0 -> Optimizers off
            1 -> Optimizers on
        """

        for pack in self._optimizer_packs.values():
            if pack.process.is_alive():
                if on_off == 1:
                    pack.allow_run_event.set()
                else:
                    pack.allow_run_event.clear()

    def _setup_system_monitoring(self):
        """ Configures the psutil monitoring of the optimization and produces a system info log message. """

        self._process = psutil.Process()
        self._process.cpu_percent()  # First return is zero and must be ignored
        psutil.getloadavg()

        cores = self._process.cpu_affinity()
        self.logger.info(f"System Info:\n"
                         f"    {'Cores Available:':.<26}{len(cores)}\n"
                         f"    {'Core IDs:':.<26}{cores}\n"
                         f"    {'Memory Available:':.<26}{present_memory(psutil.virtual_memory().total)}\n"
                         f"    {'Hostname:':.<26}{socket.gethostname()}\n"
                         f"    {'Working Dir:':.<26}{Path.cwd()}\n"
                         f"    {'Username:':.<26}{getpass.getuser()}")

    def _purge_old_results(self):
        """ Identifies and removes old log files if allowed. """

        to_remove = [*self.working_dir.glob("glompo_manager_log.yml")]
        to_remove += [*self.working_dir.glob("opt_best_summary.yml")]
        to_remove += [*self.working_dir.glob("trajectories*.png")]
        to_remove += [*self.working_dir.glob("opt*_parms.png")]
        if to_remove:
            if self.overwrite_existing:
                self.logger.debug("Old results found")
                for old in to_remove:
                    old.unlink()
                shutil.rmtree(self.working_dir / "glompo_optimizer_logs", ignore_errors=True)
                shutil.rmtree(self.working_dir / "glompo_optimizer_printstreams", ignore_errors=True)
                self.logger.warning("Deleted old results.")
            else:
                raise FileExistsError("Previous results found. Remove, move or rename them. Alternatively, select "
                                      "another working_dir or set overwrite_existing=True.")

    def _build_status_message(self) -> str:
        """ Constructs and returns a formatted status message about the optimization progress. """

        self.last_status = time()
        processes = []
        f_best = f'{self.result.fx:.3E}' if self.result.fx is not None else None
        live_opts_status = ""

        for opt_id, pack in sorted(self._optimizer_packs.items()):
            if pack.process.is_alive():
                processes.append(pack.slots)
                hist = self.opt_log.get_history(opt_id, 'fx')
                if len(hist) > 0:
                    width = 21 if hist[-1] < 0 else 22
                    live_opts_status += f"        {f'Optimizer {opt_id}':.<{width}} {hist[-1]:.3E}\n"

        evals = f"{self.f_counter:,}".replace(',', ' ')
        status_mess = f"Status: \n" \
                      f"    {'Time Elapsed:':.<26} {timedelta(seconds=time() - self.t_start)}\n" \
                      f"    {'Optimizers Alive:':.<26} {len(processes)}\n" \
                      f"    {'Slots Filled:':.<26} {sum(processes)}/{self.max_jobs}\n" \
                      f"    {'Function Evaluations:':.<26} {evals}\n" \
                      f"    Current Optimizer f_vals:\n"
        status_mess += live_opts_status
        status_mess += f"    {'Overall f_best:':.<25} {f_best}\n"

        if HAS_PSUTIL:
            with self._process.oneshot():
                self.cpu_history.append(self._process.cpu_percent())

                mem = self._process.memory_full_info().uss
                for child in self._process.children(recursive=True):
                    try:
                        mem += child.memory_full_info().uss
                    except psutil.NoSuchProcess:
                        continue
                self.mem_history.append(mem)

            status_mess += f"    {'CPU Usage:':.<26} {self.cpu_history[-1]}%\n"
            status_mess += f"    {'Virtual Memory:':.<26} {present_memory(self.mem_history[-1])}\n"
            self.load_history.append(psutil.getloadavg())
            status_mess += f"    {'System Load:':.<26} {self.load_history[-1]}\n"

        return status_mess

    def _summarise_resource_usage(self) -> Dict[str, Union[float, Sequence[float]]]:
        """ Constructs averages and standard deviation of the memory, CPU and system load statistics logged during
            the optimization.
        """

        # Verbose forcing of float and list below needed to stop recursion errors during python dump
        if len(self.load_history) > 0 and not np.all(np.isnan(self.load_history)):
            load_ave = \
                np.round(
                    np.nanmean(
                        np.reshape(
                            np.array(self.load_history, dtype=float),
                            (-1, 3)),
                        axis=0),
                    3)
            load_std = \
                np.round(
                    np.nanstd(
                        np.reshape(
                            np.array(self.load_history, dtype=float),
                            (-1, 3)),
                        axis=0),
                    3)

            load_ave = [float(i) for i in load_ave]
            load_std = [float(i) for i in load_std]
        else:
            load_ave = [0]
            load_std = [0]

        if len(self.mem_history) > 0 and not np.all(np.isnan(self.mem_history)):
            mem_max = present_memory(float(np.nanmax(self.mem_history)))
            mem_ave = present_memory(float(np.nanmean(self.mem_history)))
        else:
            mem_max = '--'
            mem_ave = '--'

        if len(self.cpu_history) > 0 and not np.all(np.isnan(self.cpu_history)):
            cpu_ave = float(np.round(np.nanmean(self.cpu_history), 2))
            cpu_std = float(np.round(np.nanstd(self.cpu_history), 2))
        else:
            cpu_ave = 0
            cpu_std = 0

        return {'load_ave': load_ave, 'load_std': load_std,
                'mem_ave': mem_ave, 'mem_max': mem_max,
                'cpu_ave': cpu_ave, 'cpu_std': cpu_std}
