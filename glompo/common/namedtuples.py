""" Named tuples used throughout the package to make code clearer. """

from multiprocessing import Event, Process
from multiprocessing.connection import Connection
from threading import Thread
from typing import Any, Callable, Dict, NamedTuple, Sequence, Type, Union

__all__ = ("Result",
           "Bound",
           "OptimizerPackage",
           "IterationResult",
           "ProcessPackage",
           "OptimizerCheckpoint")


class Result(NamedTuple):
    """ Final result delivered by GloMPOManager. """
    x: Sequence[float]
    """ Sequence[float]: Location in input space producing the lowest function value. """
    fx: float
    """ float: The lowest function value found. """
    stats: Dict[str, Any]
    """ Dict[str, Any]: Dictionary of statistics surrounding the optimization. """
    origin: Dict[str, Any]
    """ Dict[str, Any]: Optimizer name, settings, starting point and termination condition. """


class Bound(NamedTuple):
    """ Class of parameter bounds. """
    min: float
    """ float: Lower parameter bound. """
    max: float
    """ float: Upper parameter bound. """


class OptimizerPackage(NamedTuple):
    """ Package of an initialized optimizer, its unique identifier and multiprocessing variables. """
    opt_id: int
    """ int: Unique optimizer identification number. """
    optimizer: Callable
    """ Callable: Instance of :class:`.BaseOptimizer` """
    call_kwargs: Dict[str, Any]
    """ Dict[str, Any]: Dictionary of kwargs send to :meth:`.BaseOptimizer.minimize` """
    signal_pipe: Connection
    """ :class:`~multiprocessing.connection.Connection`: Used to send messages between the optimizer and manager. """
    allow_run_event: Event
    """ :class:`multiprocessing.Event`: Used by the manager to pause the optimizer. """
    slots: int
    """ int: Maximum number of thread/processes the optimizer may spawn. """


class IterationResult(NamedTuple):
    """ Return type of each optimizer iteration. """
    opt_id: int
    """ int: Unique optimizer identification number. """
    x: Sequence[float]
    """ Sequence[float]: Location in input space. """
    fx: float
    """ float: Corresponding function evaluation. """
    extras: Sequence[Any]
    """ Sequence[Any]: Possible returns if :meth:`~.BaseFunction.detailed_call` is used. """


class ProcessPackage(NamedTuple):
    """ Package of a running process and its communication channels. """
    process: Union[Process, Thread]
    """ Union[:class:`~multiprocessing.Process`, :class:`~threading.Thread`]: Process within which optimizer is run. """
    signal_pipe: Connection
    """ :class:`~multiprocessing.connection.Connection`: Used to send messages between the optimizer and manager. """
    allow_run_event: Event
    """ :class:`multiprocessing.Event`: Used by the manager to pause the optimizer. """
    slots: int
    """ int: Maximum number of thread/processes the optimizer may spawn. """


class OptimizerCheckpoint(NamedTuple):
    """ Information needed in the manager about initialized optimizers for checkpoint loading. """
    opt_type: Type['BaseOptimizer']
    """ Type[:class:`.BaseOptimizer`]: Class of optimizer to be restarted. """
    slots: int
    """ int: Maximum number of thread/processes the optimizer may spawn. """
