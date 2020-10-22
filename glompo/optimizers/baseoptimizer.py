"""
Base class from which all optimizers must inherit in order to be compatible with GloMPO.
"""

import logging
import traceback
import warnings
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from queue import Full, Queue
from threading import Event
from typing import Callable, Optional, Sequence, Tuple

from ..common.helpers import LiteralWrapper

__all__ = ('BaseOptimizer', 'MinimizeResult')


class MinimizeResult:
    """
    This class is the return value of
       * :meth:`Baseoptimizer.minimize() <scm.params.optimizers.base.BaseOptimizer.minimize>`
       * :meth:`ParameterOptimization.optimize() <scm.params.core.parameteroptimization.ParameterOptimization.optimize>`

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


class BaseOptimizer(ABC):
    """ Base class of parameter optimizers in GloMPO """

    @classmethod
    def checkpoint_load(cls, path: str, opt_id: Optional[int] = None, signal_pipe: Optional[Connection] = None,
                        results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                        backend: str = 'threads') -> 'BaseOptimizer':
        """ Recreates a previous instance of the optimizer suitable to continue a optimization from its previous
            state.

            Parameters
            ----------
            path: str
                Path to checkpoint file from which to build from. It must be a file produced by the corresponding
                BaseOptimizer().checkpoint_save method.
            opt_id, signal_pipe, results_queue, pause_flag, workers, backend
                These parameters are the same as the corresponding ones in BaseOptimizer.__init__. These will be
                regenerated and supplied by the manager during reconstruction.
        """
        raise NotImplementedError

    def __init__(self, opt_id: Optional[int] = None, signal_pipe: Optional[Connection] = None,
                 results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                 backend: str = 'threads', **kwargs):
        """
        Parameters
        ----------
        opt_id: Optional[int] = None
            Unique identifier automatically assigned within the GloMPO manager framework.
        signal_pipe: Optional[multiprocessing.connection.Connection] = None
            Bidirectional pipe used to message management behaviour between the manager and optimizer.
        results_queue: Optional[queue.Queue] = None
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
        kwargs
            Optimizer specific initialization arguments.
        """
        self.logger = logging.getLogger(f'glompo.optimizers.opt{opt_id}')
        self._opt_id = opt_id
        self._signal_pipe = signal_pipe
        self._results_queue = results_queue
        self._pause_signal = pause_flag  # If set allow run, if cleared wait.
        self._backend = backend
        self._result_cache = None

        self._FROM_MANAGER_SIGNAL_DICT = {0: self.checkpoint_save,
                                          1: self.callstop,
                                          2: self._prepare_checkpoint}
        self._TO_MANAGER_SIGNAL_DICT = {0: "Normal Termination",
                                        1: "Confirm Pause",
                                        9: "Other Message (Saved to Log)"}
        self.workers = workers

    def _minimize(self,
                  function: Callable[[Sequence[float]], float],
                  x0: Sequence[float],
                  bounds: Sequence[Tuple[float, float]],
                  callbacks: Callable = None, **kwargs) -> MinimizeResult:
        """ Wrapper around minimize that captures KeyboardInterrupt exceptions to exit gracefully and
            other Exceptions to log them.
        """
        try:
            return self.minimize(function, x0, bounds, callbacks, **kwargs)
        except (KeyboardInterrupt, BrokenPipeError):
            print("Interrupt signal received. Process stopping.")
            self.logger.warning("Interrupt signal received. Process stopping.")
        except Exception as e:
            formatted_e = "".join(traceback.TracebackException.from_exception(e).format())
            self.logger.critical("Critical error encountered", exc_info=e)
            self._signal_pipe.send((9, LiteralWrapper(formatted_e)))
            raise e

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
            - Must include a call to put every iteration in the results_queue via the push_iter_result method.
            - Each deposit into the results queue must be an instance of the IterationResult NamedTuple.
            - Messages to the GloMPO manager must be sent via the message_manager method. The keys for these messages
              are detailed in the _TO_MANAGER_SIGNAL_DICT dictionary.
            - A convergence termination message should be sent if the optimizer successfully converges.
            - Messages from the manager can be read by the check_messages method. See the _FROM_MANAGER_SIGNAL_DICT
              for all the methods which must be implemented to interpret GloMPO signals correctly.
            - self._pause_signal.wait() must be implemented in the body of the iterative loop to allow the optimizer to
              be paused by the manager as needed.
            - The TestSubclassGlompoCompatible test in test_optimizers.py can be used to test that an optimizer meets
              these criteria and is GloMPO compatible.


        Example:

        .. code-block:: python

            while not self.stop:
                self.optimize()     # Do one optimization step
                if callbacks and callbacks():     # Stop if callbacks() returns `True`
                    self.callstop('Callbacks returned True')


        :Returns: An instance of |MinimizeResult|

        """

    def check_messages(self, *args):
        while self._signal_pipe.poll():
            message = self._signal_pipe.recv()
            self.logger.debug(f"Received signal: {message}")
            if isinstance(message, int) and message in self._FROM_MANAGER_SIGNAL_DICT:
                self.logger.debug(f"Executing: {self._FROM_MANAGER_SIGNAL_DICT[message].__name__}")
                self._FROM_MANAGER_SIGNAL_DICT[message]()
            elif isinstance(message, tuple) and message[0] in self._FROM_MANAGER_SIGNAL_DICT:
                self.logger.debug(f"Executing: {self._FROM_MANAGER_SIGNAL_DICT[message[0]].__name__}")
                self._FROM_MANAGER_SIGNAL_DICT[message[0]](*message[1:])
            else:
                self.logger.warning("Cannot parse message, ignoring")
                warnings.warn("Cannot parse message, ignoring", RuntimeWarning)

    def push_iter_result(self, result: 'IterationResult'):
        """ Put an iteration result into _results_queue.
            Will block until the result is passed to the queue but does timeout every 1s to process any messages from
            the manager. Should not be overwritten.
        """
        self._result_cache = result
        while self._result_cache:
            try:
                self._results_queue.put(result, block=True, timeout=1)
                self._result_cache = None
            except Full:
                self.check_messages()

    def message_manager(self, key: int, message: Optional[str] = None):
        self._signal_pipe.send((key, message))

    def callstop(self, *args):
        """
        Signal to terminate the :meth:`minimize` loop while still returning a result
        """
        raise NotImplementedError

    def checkpoint_save(self, path: str):
        """ Save current state, suitable for restarting. Path is the location for the file or folder to be constructed.
            Note that only the absolutely critical aspects of the state of the optimizer need to be saved. The manager
            will resupply multiprocessing parameters when the optimizer is reconstructed.
        """
        raise NotImplementedError

    def _prepare_checkpoint(self):
        """ Process to pause, synchronize and save optimizers. Should not be overwritten. """
        self.logger.debug("Preparing for Checkpoint")
        if self._result_cache:
            self.logger.debug("Outstanding result found. Pushing to queue...")
            self._results_queue.put(self._result_cache, block=True)
            self.logger.debug(f"Oustanding result (iter={self._result_cache.n_iter}) pushed")
            self._result_cache = None
        self.message_manager(1)  # Certify waiting for next instruction
        self.logger.debug("Wait signal messaged to manager, waiting for reply...")
        self._signal_pipe.poll(timeout=None)  # Wait on instruction to save or end
        self.logger.debug("Instruction received. Executing...")
        self.check_messages()
        self.logger.debug("Instructions processed. Pausing until release...")
        self._pause_signal.clear()  # Wait on pause event, to be released by manager
        self._pause_signal.wait()
        self.logger.debug("Pause released by manager. Checkpointing completed.")
