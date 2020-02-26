"""
Base class from which all optimizers must inherit in order to be compatible with GloMPO.
"""

from multiprocessing.connection import Connection
from multiprocessing import Queue
from typing import *
from abc import ABC, abstractmethod


__all__ = ['BaseOptimizer', 'MinimizeResult']


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


class BaseOptimizer(ABC):
    """Base class of parameter optimizers in ParAMS.

    Classes representing specific optimizers (e.g. |OptCMA|) can derive from this abstract base class.

    Attributes:

    needscaler : `bool`
        Whether the optimizer requires parameter scaing or not.

        .. warning:: This variable **must** be defined with every optimizer.


    """

    @property
    @classmethod
    @abstractmethod
    def needscaler(cls):
        """
        Force class variable 'needscaler'. To be set to True or False
        """
        pass

    def __init__(self, opt_id: int = None, signal_pipe: Connection = None, results_queue: Queue = None):
        self.__opt_id = opt_id
        self.__signal_pipe = signal_pipe
        self.__results_queue = results_queue
        self.__SIGNAL_DICT = {0: self.save_state,
                              1: self.callstop}

    @abstractmethod
    def minimize(self,
                 function: Callable,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        # TODO Expand description of results_queue and signal_pipe
        """
        Minimizes a function, given an initial list of variable values `x0`, and possibly a list of `bounds` on the
        variable values. The `callbacks` argument allows for specific callbacks such as early stopping.

        NB Must include a call to put every iteration in the results_queue. Messages to the mp_manager must be sent via the
        message_manager method. Messages from the mp_manager can be read by the check_messages method. For proper GloMPO
        functionality these must all be implemented appropriately in this method.

        Example:

        .. code-block:: python

            while not self.stop:
                self.optimize()     # Do one optimization step
                if callbacks and callbacks():     # Stop if callbacks() returns `True`
                    self.callstop('Callbacks returned True')


        :Returns: An instance of |MinimizeResult|

        """
        pass

    def check_messages(self, *args):
        while self.__signal_pipe.poll():
            code, sig_args = self.__signal_pipe.recv()
            self.__SIGNAL_DICT[code](*sig_args)

    def message_manager(self, *args):
        pass

    def callstop(self, *args):
        """
        Signal to terminate the :meth:`minimize` loop while still returning a result
        """
        pass

    def save_state(self, *args):
        pass