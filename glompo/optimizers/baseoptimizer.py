"""
Base class from which all optimizers must inherit in order to be compatible with GloMPO.
"""

import logging
import traceback
import warnings
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Event
from typing import Callable, List, Optional, Sequence, Set, Tuple, Type, Union

from ..common.helpers import LiteralWrapper
from ..common.namedtuples import IterationResult
from ..common.wrappers import needs_optional_package
from ..core._backends import ChunkingQueue

try:
    import dill
except ModuleNotFoundError:
    pass

__all__ = ('BaseOptimizer', 'MinimizeResult')


class MinimizeResult:
    """ The return value of :class:`BaseOptimizer` classes.
    The results of an optimization can be accessed by:

    Attributes
    ----------
    success : bool
        Whether the optimization was successful or not.
    x : Sequence[float]
        The optimized parameters.
    fx : float
        The corresponding function value of :obj:`x`.
    stats : Dict[str, Any]
        Dictionary of various statistics related to the optimization.
    origin : Dict[str, Any]
        Dictionary with configurations details of the optimizer which produced the result.
    """

    def __init__(self):
        self.success = False
        self.x = None
        self.fx = float('inf')
        self.stats = None
        self.origin = None


class _MessagingWrapper:
    """ Messages results to the manager whenever the optimization task is evaluated.
    Automatically wrapped around the optimization task by the GloMPO manager for each optimizer it starts.

    Parameters
    ----------
    func
        Function to be minimized.
    results_queue
        Results queue into which results are put.
    opt_id
        Optimizer to which this wrapper is associated.
    is_log_detailed
        If :obj:`True`, using :meth:`__call__` will log the return of :meth:`detailed_call` in the log.
        If :obj:`False`, using :meth:`__call__` will log the return of :meth:`__call__` in the log.
    """

    def __init__(self,
                 func: Callable[[Sequence[float]], float],
                 results_queue: ChunkingQueue,
                 opt_id: int,
                 is_log_detailed: bool):
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
    """ Abstract base class for optimizers used within the GloMPO framework.
    Cannot be used directly, must be superclassed by child classes which implement a specific optimization algorithm.

    .. attention::

       To Ensure GloMPO Functionality:

       #. Messages to the GloMPO manager must be sent via :meth:`message_manager`.

       #. Messages from the manager must be read by :meth:`check_messages` which executes :class:`BaseOptimizer`
          methods corresponding to the signals. The defaults provided in the :class:`BaseOptimizer` class are generally
          suitable and should not need to be overwritten! The only methods which must implemented by the user are:

             #. :meth:`minimize` which is the algorithm specific optimization loop;

             #. :meth:`callstop` which interrupts the optimization loop.

       #. The statement :code:`self._pause_signal.wait()` must appear somewhere in the body of the iterative loop to
          allow the optimizer to be paused by the manager as needed.

       #. Optional: the class should be able to handle resuming an optimization from any point using
          :meth:`checkpoint_save` and :meth:`checkpoint_load`.

    .. tip::

       The :code:`TestSubclassGlompoCompatible` test in ``test_optimizers.py`` can be used to test that an optimizer
       meets these criteria and is GloMPO compatible. Simply add your optimizer to :code:`AVAILABLE_CLASSES` there.

    Parameters
    ----------
    _opt_id
        Unique optimizer identifier.

    _signal_pipe
        Bidirectional pipe used to message management behaviour between the manager and optimizer.

    _results_queue
        Threading queue into which optimizer iteration results are centralised across all optimizers and sent to
        the manager.

    _pause_flag
        Event flag which can be used to pause the optimizer between iterations.

    _is_log_detailed
        See :attr:`is_log_detailed`.

    workers
        The number of concurrent calculations used by the optimizer. Defaults to one. The manager will only start
        the optimizer if there are sufficient slots available for it.

    backend
        The type of concurrency used by the optimizers (processes or threads). This is not necessarily applicable to
        all optimizers. This will default to :code:`'threads'` unless forced to use :code:`'processes'` (see
        :meth:`.GloMPOManager.setup` and :ref:`Parallelism`).

    **kwargs
        Optimizer specific initialization arguments.

    Notes
    -----
    The user need not concern themselves with the particulars of the `_opt_id`, `_signal_pipe`, `_results_queue`
    and `_pause_flag` parameters. These are automatically generated by the manager.

    .. important::

       Make sure to call the superclass initialization method when creating your own optimizers:

          super().__init__(_opt_id,
                           _signal_pipe,
                           _results_queue,
                           _pause_flag,
                           _is_log_detailed
                           workers,
                           backend)

    Attributes
    ----------
    incumbent : Dict[str, Any]
        Dictionary with keys :code:`'x'` and :code:`'fx'` which contain the lowest function value and associated
        parameter vector seen thus far by the optimizer.
    is_log_detailed : bool
        If :obj:`True`:

           #. When the task's :meth:`~.BaseFunction.__call__` method is called, its :meth:`~.BaseFunction.detailed_call`
              method will actually be evaluated.

           #. All the return values from :meth:`~.BaseFunction.detailed_call` will be added to the log history of the
              optimizer.

           #. The function itself will only return the function value (as if the :meth:`~.BaseFunction.__call__` method
              had been used).

        .. note::

           This will *not* result in a doubling of the computational time as the original call will be intercepted.
           This setting is useful for cases where optimizers do not need/cannot handle the extra information generated
           by a detailed call but one would still like the iteration details logged for analysis.

    logger : logging.Logger
        :class:`logging.Logger` instance into which status messages may be added.
    workers : int
        Maximum number of threads/processes the optimizer may use for evaluating the objective function.
    """

    @property
    def is_restart(self):
        """ :obj:`True` if the optimizer is loaded from a checkpoint. """
        return self._is_restart

    @property
    def opt_id(self):
        """ The unique GloMPO generated identification number of the optimizer. """
        return self._opt_id

    @classmethod
    @needs_optional_package('dill')
    def checkpoint_load(cls: Type['BaseOptimizer'], path: Union[Path, str], **kwargs) -> 'BaseOptimizer':
        """ Recreates an optimizer from a saved snapshot.

        Parameters
        ----------
        path
            Path to checkpoint file from which to build from. It must be a file produced by the corresponding
            :meth:`checkpoint_save` method.
        **kwargs
            See :class:`__init__ <.BaseOptimizer>`.

        Notes
        -----
        This is a basic implementation which should suit most optimizers; may need to be overwritten.
        """
        opt = cls.__new__(cls)
        super(cls, opt).__init__(**kwargs)

        with open(path, 'rb') as file:
            state = dill.load(file)

        for var, val in state.items():
            opt.__setattr__(var, val)
        opt._is_restart = True

        opt.logger.info("Successfully loaded from restart file.")
        return opt

    def __init__(self,
                 _opt_id: Optional[int] = None,
                 _signal_pipe: Optional[Connection] = None,
                 _results_queue: Optional[ChunkingQueue] = None,
                 _pause_flag: Optional[Event] = None,
                 _is_log_detailed: bool = False,
                 workers: int = 1,
                 backend: str = 'threads',
                 **kwargs):
        self.logger = logging.getLogger(f'glompo.optimizers.opt{_opt_id}')
        self._opt_id = _opt_id
        self._signal_pipe = _signal_pipe
        self._results_queue = _results_queue
        self._pause_signal = _pause_flag  # If set allow run, if cleared wait.
        self._backend = backend
        self._is_restart = False

        self._from_manager_signal_dict = {0: self.checkpoint_save,
                                          1: self.callstop,
                                          2: self._prepare_checkpoint,
                                          3: self._checkpoint_pass,
                                          4: self.inject}
        self.workers = workers
        self.incumbent = {'x': None, 'fx': None}
        self.is_log_detailed = _is_log_detailed

    @abstractmethod
    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        """ Run the optimization algorithm to minimize a function.

        Parameters
        ----------
        function
            Function to be minimised. See :class:`.BaseFunction` for an API guide.

        x0
            The initial optimizer starting point.

        bounds
            Min/max boundary limit pairs for each element of the input vector to the minimisation function.

        callbacks
            Code snippets usually called once per iteration that are able to signal early termination. Callbacks are
            leveraged differently by different optimizer implementations, the user is encouraged to consult the child
            classes for more details. Use of callbacks, however, is *strongly discouraged*.
        """

    def check_messages(self) -> List[int]:
        """ Processes and executes manager signals from the manager.

        .. danger::

           This implementation has been very carefully structured to operate as expected by the manager. Should be
           suitable for all optimizers. **Should not** be overwritten.

        Returns
        -------
        List[int]
            Signal keys received by the manager during the call.
        """
        processed_signals = []
        while self._signal_pipe.poll():
            message = self._signal_pipe.recv()
            self.logger.debug("Received signal: %s", message)
            if isinstance(message, int) and message in self._from_manager_signal_dict:
                self.logger.debug("Executing: %s", self._from_manager_signal_dict[message].__name__)
                processed_signals.append(message)
                self._from_manager_signal_dict[message]()
            elif isinstance(message, tuple) and message[0] in self._from_manager_signal_dict:
                processed_signals.append(message[0])
                self.logger.debug("Executing: %s", self._from_manager_signal_dict[message[0]].__name__)
                self._from_manager_signal_dict[message[0]](*message[1:])
            else:
                self.logger.warning("Cannot parse message, ignoring")
                warnings.warn("Cannot parse message, ignoring", RuntimeWarning)

        return processed_signals

    def message_manager(self, key: int, message: Optional[str] = None):
        """ Sends arguments to the manager.

        .. caution::

           Should not be overwritten.

        Parameters
        ----------
        key
            Indicates the type of signal sent. The manager recognises the following keys:

            0: The optimizer has terminated normally according to its own internal convergence conditions.

            1: Confirm that a pause signal has been received from the manager and the optimizer has complied with the
            request.

            9: General message to be appended to the optimizer's log.
        message
            Message to be appended when sending signal 9.
        """
        self._signal_pipe.send((key, message))

    @abstractmethod
    def callstop(self, reason: str):
        """ Breaks out of the :meth:`minimize` minimization loop. """

    @needs_optional_package('dill')
    def checkpoint_save(self, path: Union[Path, str], force: Optional[Set[str]] = None):
        """ Save current state, suitable for restarting.

        Parameters
        ----------
        path
            Path to file into which the object will be dumped. Typically supplied by the manager.
        force
            Set of variable names which will be forced into the dumped file. Convenient shortcut for overwriting if
            fails for a particular optimizer because a certain variable is filtered out of the data dump.

        Notes
        -----
        #. Only the absolutely critical aspects of the state of the optimizer need to be saved. The manager will
           resupply multiprocessing parameters when the optimizer is reconstructed.

        #. This method will almost never be called directly by the user. Rather it will called (via signals) by the
           manager.

        #. This is a basic implementation which should suit most optimizers; may need to be overwritten.
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
        """ Updates the :attr:`incumbent` with a better solution from the manager. """
        self.incumbent = {'x': x, 'fx': fx}

    def _checkpoint_pass(self):
        """ Allows optimizers captured by checkpoint to pass out without saving.

        .. caution::

           Empty method. Should not be overwritten.
        """

    def _prepare_checkpoint(self):
        """ Process to pause, synchronize and save optimizers.

        .. caution::

           Should not be overwritten.
        """
        self.logger.debug("Preparing for Checkpoint")
        self.message_manager(1)  # Certify waiting for next instruction
        self.logger.debug("Wait signal messaged to manager, waiting for reply...")

        processed_signals = []
        while all([s not in processed_signals for s in (0, 3)]):
            self._signal_pipe.poll(timeout=None)  # Wait on instruction to save or end
            self.logger.debug("Instruction received. Executing...")
            processed_signals = self.check_messages()
        self.logger.debug("Instructions processed. Pausing until release...")
        self._pause_signal.clear()  # Wait on pause event, to be released by manager
        self._pause_signal.wait()
        self.logger.debug("Pause released by manager. Checkpointing completed.")

    def _minimize(self,
                  function: Callable[[Sequence[float]], float],
                  x0: Sequence[float],
                  bounds: Sequence[Tuple[float, float]],
                  callbacks: Callable = None, **kwargs) -> MinimizeResult:
        """ Wrapper around :meth:`minimize` which adds GloMPO specific functionality.
        Main purposes are to:

        #. Wrap the function with :class:`_MessagingWrapper`;

        #. Capture :exc:`KeyboardInterrupt` exceptions to exit gracefully;

        #. Capture other :exc:`Exception`s to log them.

        #. Correctly handles the opening and closing of the optimizer log file if it is being constructed.

        .. warning::

           Do not overwrite.
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
