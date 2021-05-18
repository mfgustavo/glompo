"""
Base class from which all optimizers must inherit in order to be compatible with GloMPO.
"""

import logging
import traceback
import warnings
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Type, Union

import dill

from ..common.helpers import LiteralWrapper
from ..common.namedtuples import IterationResult
from ..core._backends import ChunkingQueue

__all__ = ('BaseOptimizer', 'MinimizeResult')


class MinimizeResult:
    """
    This class is the return value of BaseOptimizer classes.
    The results of an optimization can be accessed by:

    Attributes:

    success : bool
        Whether the optimization was successful or not
    x : numpy.array
        The optimized parameters
    fx : float
        The corresponding |Fitfunc| value of `x`
    """

    def __init__(self):
        self.success = False
        self.x = None
        self.fx = float('inf')
        self.stats = None
        self.origin = None


class _MessagingWrapper:
    """ Core functionality which messages evaluations to the manager from the children.
        Automatically wrapped around the optimization task by each optimizer.
    """

    def __init__(self,
                 func: Callable[[Sequence[float]], float],
                 results_queue: ChunkingQueue,
                 opt_id: int,
                 is_log_detailed: bool):
        """ Initialises the wrapper class.

            Parameters
            ----------
            func: Callable[[Sequence[float]], float]
                Function to be minimized.
            results_queue: ChunkingQueue
                Results queue into which results are put.
            opt_id: int
                Optimizer to which this wrapper is associated.
            is_log_detailed: bool
                If True, using __call__ will log the return of detailed_call in the log.
                If False, using __call__ will log the return of call in the log.
        """
        self.func = func
        self.results_queue = results_queue
        self.opt_id = opt_id
        self.is_log_detailed = is_log_detailed

        if is_log_detailed and not hasattr(self.func, 'detailed_call'):
            raise AttributeError("func does not have 'detailed_call' method")

    def __call__(self, x: Sequence[float]) -> float:
        return self._calculate(x, caller='call')

    def detailed_call(self, x: Sequence[float]) -> Sequence:
        return self._calculate(x, caller='detailed_call')

    def _calculate(self, x: Sequence[float], caller: str) -> Union[float, Sequence]:
        is_det_call = self.is_log_detailed or caller == 'detailed_call'
        if is_det_call:
            calc = self.func.detailed_call(x)
        else:
            calc = (self.func(x),)

        result = IterationResult(opt_id=self.opt_id,
                                 x=x,
                                 fx=calc[0],
                                 extras=calc[1:])
        self.results_queue.put_nowait(result)

        if caller == 'call':
            return calc[0]
        return calc


class BaseOptimizer(ABC):
    """ Base class of parameter optimizers in GloMPO """

    @classmethod
    def checkpoint_load(cls: Type['BaseOptimizer'],
                        path: Union[Path, str],
                        opt_id: Optional[int] = None,
                        signal_pipe: Optional[Connection] = None,
                        results_queue: Optional[Queue] = None,
                        pause_flag: Optional[Event] = None,
                        workers: int = 1,
                        backend: str = 'threads',
                        is_log_detailed: bool = False) -> 'BaseOptimizer':
        """ Recreates a previous instance of the optimizer suitable to continue a optimization from its previous
            state. Below is a basic implementation which should suit most optimizers, may need to be overwritten.

            Parameters
            ----------
            path: Union[Path, str]
                Path to checkpoint file from which to build from. It must be a file produced by the corresponding
                BaseOptimizer().checkpoint_save method.
            opt_id, signal_pipe, results_queue, pause_flag, workers, backend
                These parameters are the same as the corresponding ones in BaseOptimizer.__init__. These will be
                regenerated and supplied by the manager during reconstruction.
        """
        opt = cls.__new__(cls)
        super(cls, opt).__init__(opt_id, signal_pipe, results_queue, pause_flag,
                                 workers, backend, is_log_detailed)

        with open(path, 'rb') as file:
            state = dill.load(file)

        for var, val in state.items():
            opt.__setattr__(var, val)
        opt._is_restart = True

        opt.logger.info("Successfully loaded from restart file.")
        return opt

    @property
    def is_restart(self):
        return self._is_restart

    @property
    def opt_id(self):
        return self._opt_id

    def __init__(self,
                 opt_id: Optional[int] = None,
                 signal_pipe: Optional[Connection] = None,
                 results_queue: Optional[ChunkingQueue] = None,
                 pause_flag: Optional[Event] = None,
                 workers: int = 1,
                 backend: str = 'threads',
                 is_log_detailed: bool = False, **kwargs):
        """
        Initialisation of the base optimizer. Must be called by any child classes.

        Parameters
        ----------
        opt_id: Optional[int] = None
            Unique identifier automatically assigned within the GloMPO manager framework.

        signal_pipe: Optional[multiprocessing.connection.Connection] = None
            Bidirectional pipe used to message management behaviour between the manager and optimizer.

        results_queue: Optional[ChunkingQueue] = None
            Threading queue into which optimizer iteration results are centralised across all optimizers and sent to
            the manager.

        pause_flag: Optional[threading.Event] = None
            Event flag which can be used to pause the optimizer between iterations.

        workers: int = 1
            The number of concurrent calculations used by the optimizer. Defaults to one. The manager will only start
            the optimizer if there are sufficient slots available for it:
                workers <= manager.max_jobs - manager.n_slots_occupied.

        backend: str = 'threads'
            The type of concurrency used by the optimizers (processes or threads). This is not necessarily applicable to
            all optimizers. This will default to threads unless forced to used processes (see GloMPOManger backend
            argument for details).

        is_log_detailed: bool = False
            If True, a detailed_call will be run and logged everytime a normal function call is made. Note that this
            will NOT result in a doubling of the computational time as the original call will be intercepted. This
            setting is useful for cases where optimizers do not need/cannot handle the extra information generated by a
            detailed call but one would still like the iteration details logged for analysis.
            See BaseOptimizer.minimize docstring for more information on detailed calls.

        kwargs
            Optimizer specific initialization arguments.
        """
        self.logger = logging.getLogger(f'glompo.optimizers.opt{opt_id}')
        self._opt_id = opt_id
        self._signal_pipe = signal_pipe
        self._results_queue = results_queue
        self._pause_signal = pause_flag  # If set allow run, if cleared wait.
        self._backend = backend
        self._is_restart = False

        self._FROM_MANAGER_SIGNAL_DICT = {0: self.checkpoint_save,
                                          1: self.callstop,
                                          2: self._prepare_checkpoint,
                                          3: self._checkpoint_pass,
                                          4: self.inject}
        self._TO_MANAGER_SIGNAL_DICT = {0: "Normal Termination",
                                        1: "Confirm Pause",
                                        9: "Other Message (Saved to Log)"}
        self.workers = workers
        self.incumbent = {'x': None, 'fx': None}
        self.is_log_detailed = is_log_detailed

    @abstractmethod
    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        """
        Minimizes a function, given an initial list of variable values `x0`, and possibly a list of `bounds` on the
        variable values. The `callbacks` argument allows for specific callbacks such as early stopping.

        NB
            To Ensure GloMPO Functionality:
            - Messages to the GloMPO manager must be sent via the message_manager method. The keys for these messages
              are detailed in the _TO_MANAGER_SIGNAL_DICT dictionary.
            - A convergence termination message should be sent if the optimizer successfully converges.
            - Messages from the manager can be read by the check_messages method. See the _FROM_MANAGER_SIGNAL_DICT
              for all the methods which must be implemented to interpret GloMPO signals correctly. The defaults are
              generally suitable, however, and do not need to be overwritten! The only requirement is the implementation
              of the callstop method which interrupts the optimization loop.
            - self._pause_signal.wait() must be implemented in the body of the iterative loop to allow the optimizer to
              be paused by the manager as needed.
            - The TestSubclassGlompoCompatible test in test_optimizers.py can be used to test that an optimizer meets
              these criteria and is GloMPO compatible.
            - Should be able to handle resuming an optimization from any point using the checkpoint_save and
              checkpoint_load methods.

        Parameters
        ----------
        function: Callable[[Sequence[float]], float]
            Function to be minimised. The minimum requirement is a Callable which accepts a vector in input space and
            returns a single float.

        x0: Sequence[float]
            The initial optimizer starting point.

        bounds: Sequence[Tuple[float, float]]
            Min max boundary limit pairs for each element of the input vector to the minimisation function.

        callbacks: Callable = None
            Code snippets usually called once per iteration that are able to signal early termination. Callbacks are
            leveraged differently by different optimizer implementations, the user is encouraged to consult the child
            classes for more details.
        """

    def check_messages(self, return_signals: bool = False) -> Optional[List[int]]:
        """ Processes and executes manager signals from the manager. Should not be overwritten.

            Parameters
            -------
            return_signals: bool = False
                If True the method returns the signal keys received by the manager in the call
        """
        processed_signals = []
        while self._signal_pipe.poll():
            message = self._signal_pipe.recv()
            self.logger.debug("Received signal: %s", message)
            if isinstance(message, int) and message in self._FROM_MANAGER_SIGNAL_DICT:
                self.logger.debug("Executing: %s", self._FROM_MANAGER_SIGNAL_DICT[message].__name__)
                processed_signals.append(message)
                self._FROM_MANAGER_SIGNAL_DICT[message]()
            elif isinstance(message, tuple) and message[0] in self._FROM_MANAGER_SIGNAL_DICT:
                processed_signals.append(message[0])
                self.logger.debug("Executing: %s", self._FROM_MANAGER_SIGNAL_DICT[message[0]].__name__)
                self._FROM_MANAGER_SIGNAL_DICT[message[0]](*message[1:])
            else:
                self.logger.warning("Cannot parse message, ignoring")
                warnings.warn("Cannot parse message, ignoring", RuntimeWarning)

        if return_signals:
            return processed_signals

    def message_manager(self, key: int, message: Optional[Any] = None):
        """ Sends arguments to the manager. key indicates the type of signal sent (see _TO_MANAGER_SIGNAL_DICT) and
            message contains extra information which may be needed to process the request. Should not be overwritten.
        """
        self._signal_pipe.send((key, message))

    @abstractmethod
    def callstop(self, reason: str):
        """ Signal to terminate the minimize loop while still returning a result. """

    def checkpoint_save(self, path: Union[Path, str], force: Optional[Set[str]] = None):
        """ Save current state, suitable for restarting. Path is the location for the file or folder to be constructed.
            Note that only the absolutely critical aspects of the state of the optimizer need to be saved. The manager
            will resupply multiprocessing parameters when the optimizer is reconstructed. Below is a basic
            implementation which should suit most optimizers, may need to be overwritten.

            Parameters
            ----------
            path: Union[Path, str]
                Path to file into which the object will be dumped.
            force: Optional[str]
                Set of variable names which will be forced into the dumped file. Convenient shortcut for overwriting if
                fails for a particular optimizer because a certain variable is filtered out of the data dump.
        """
        self.logger.debug("Creating restart file.")

        force = set(force) if force else set()
        dump_collection = {}
        for var in dir(self):
            if not callable(getattr(self, var)) and \
                    not var.startswith('_') and \
                    all([var != forbidden for forbidden in ('logger', 'is_restart', 'opt_id', 'n_iter')]) or \
                    var in force:
                dump_collection[var] = getattr(self, var)
        with open(path, 'wb') as file:
            dill.dump(dump_collection, file)

        self.logger.info("Restart file created successfully.")

    def inject(self, x: Sequence[float], fx: float):
        """ If configured to do so, the manager will share the best solution seen by any optimizer with the others
            through this method. The default is to save the iteration into the incumbent property which the minimize
            algorithm may be able to use in some way.
        """
        self.incumbent = {'x': x, 'fx': fx}

    def _checkpoint_pass(self):
        """ Empty method. Allows optimizers captured by checkpoint to pass out without saving.
            Should not be overwritten.
        """

    def _prepare_checkpoint(self):
        """ Process to pause, synchronize and save optimizers. Should not be overwritten. """
        self.logger.debug("Preparing for Checkpoint")
        self.message_manager(1)  # Certify waiting for next instruction
        self.logger.debug("Wait signal messaged to manager, waiting for reply...")

        processed_signals = []
        while all([s not in processed_signals for s in (0, 3)]):
            self._signal_pipe.poll(timeout=None)  # Wait on instruction to save or end
            self.logger.debug("Instruction received. Executing...")
            processed_signals = self.check_messages(return_signals=True)
        self.logger.debug("Instructions processed. Pausing until release...")
        self._pause_signal.clear()  # Wait on pause event, to be released by manager
        self._pause_signal.wait()
        self.logger.debug("Pause released by manager. Checkpointing completed.")

    def _minimize(self,
                  function: Callable[[Sequence[float]], float],
                  x0: Sequence[float],
                  bounds: Sequence[Tuple[float, float]],
                  callbacks: Callable = None, **kwargs) -> MinimizeResult:
        """ Wrapper around minimize that captures KeyboardInterrupt exceptions to exit gracefully and
            other Exceptions to log them.

            Also correctly handles the opening and closing of the optimizer log file if it is being constructed.
        """
        try:
            function = _MessagingWrapper(function, self._results_queue, self.opt_id, self.is_log_detailed)
            return self.minimize(function, x0, bounds, callbacks, **kwargs)

        except (KeyboardInterrupt, BrokenPipeError):
            print("Interrupt signal received. Process stopping.")
            self.logger.warning("Interrupt signal received. Process stopping.")

        except Exception as e:
            formatted_e = "".join(traceback.TracebackException.from_exception(e).format())
            self.logger.critical("Critical error encountered", exc_info=e)
            self._signal_pipe.send((9, LiteralWrapper(formatted_e)))
            raise e

        finally:
            self._results_queue.put(self.opt_id)
