""" Named tuples used throughout the package to make code clearer. """

from multiprocessing import Event, Process
from multiprocessing.connection import Connection
from threading import Thread
from typing import Any, Callable, Dict, NamedTuple, Sequence, Type, Union

from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("Result",
           "Bound",
           "OptimizerPackage",
           "IterationResult",
           "ProcessPackage",
           "OptimizerCheckpoint",
           "LoggingOptions")


class Result(NamedTuple):
    """ Final result delivered by GloMPOManager. """
    x: Sequence[float]
    fx: float
    stats: Dict[str, Any]
    origin: Dict[str, Any]  # Optimizer name, settings, starting point and termination condition


class Bound(NamedTuple):
    """ Class of parameter bounds. """
    min: float
    max: float


class OptimizerPackage(NamedTuple):
    """ Package of an initialized optimizer, its unique identifier and multiprocessing variables. """
    opt_id: int
    optimizer: Callable
    call_kwargs: Dict[str, Any]
    signal_pipe: Connection
    allow_run_event: Event
    slots: int


class IterationResult(NamedTuple):
    """ Return type of each optimizer iteration. """
    opt_id: int
    n_iter: int
    i_fcalls: int  # Number of functions calls in iteration *not* cumulative calls
    x: Sequence[float]
    fx: float
    final: bool  # True if this is the final result sent to the queue by this optimizer


class ProcessPackage(NamedTuple):
    """ Package of a running process and its communication channels. """
    process: Union[Process, Thread]
    signal_pipe: Connection
    allow_run_event: Event
    slots: int


class OptimizerCheckpoint(NamedTuple):
    """ Information needed in the manager about initialized optimizers for checkpoint loading. """
    opt_type: Type[BaseOptimizer]
    slots: int


class LoggingOptions(NamedTuple):
    """ Holds GloMPO manager logging and saving options. """
    save_manager_summary: bool = True  # YAML file with summary info about the optimization and the result.
    save_optimizer_summary: bool = True  # YAML file with summary info of every optimizer started.
    save_optimizer_logs: bool = False  # CSVs with iteration history of every optimizer started.
    make_detailed_optimizer_logs: bool = False  # CSVs will log results of task.detailed_call (see BaseOptimizer).
    make_trajectory_plot: bool = True  # PNG of all explored error values v time.
    make_optimizer_plots: bool = False  # PNG for each optimizer showing parameter values tested v time.
