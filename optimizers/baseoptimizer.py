"""
Base class from which all optimizers must inherit in order to be compatible with GloMPO.
"""

import multiprocessing as mp
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

    @abstractmethod
    def minimize(self,
                 function: Callable,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 results_queue: Type[mp.Manager().Queue],
                 signal_pipe,
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        # TODO Expand description of results_queue and signal_pipe
        """
        Minimizes a function, given an initial list of variable values `x0`, and possibly a list of `bounds` on the
        variable values. The `callbacks` argument allows for specific callbacks such as early stopping.

        NB Must include a call to put every iteration in the results_queue. Messages to the manager must be sent via the
        signal_pipe

        Example:

        .. code-block:: python

            while not self.stop:
                self.optimize()     # Do one optimization step
                if callbacks and callbacks():     # Stop if callbacks() returns `True`
                    self.callstop('Callbacks returned True')


        :Returns: An instance of |MinimizeResult|

        """
        pass

    def callstop(self, reason=None):
        """
        Signal to terminate the :meth:`minimize` loop while still returning a result
        """
        pass

    def pause(self):
        pass

    def unpause(self):
        pass
