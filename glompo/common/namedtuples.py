

""" Named tuples used throughout the package to make code clearer. """


from typing import *
from multiprocessing import Process, Event
from multiprocessing.connection import Connection
from ..core.gpr import GaussianProcessRegression


__all__ = ("Result",
           "Bound",
           "OptimizerPackage",
           "IterationResult",
           "HyperparameterOptResult",
           "ProcessPackage")


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
    gpr: GaussianProcessRegression


class IterationResult(NamedTuple):
    """ Return type of each optimizer iteration. """
    opt_id: int
    n_iter: int
    n_icalls: int  # Number of functions calls in iteration *not* cumulative calls
    x: Sequence[float]
    fx: float
    final: bool  # True if this is the final result sent to the queue by this optimizer


class HyperparameterOptResult(NamedTuple):
    """ Return type of a hyperparameter optimization job. """
    opt_id: int
    alpha: float
    beta: float
    sigma: float


class ProcessPackage(NamedTuple):
    """ Package of a running process, its communication channels and GPR. """
    process: Process
    signal_pipe: Connection
    allow_run_event: Event
    gpr: GaussianProcessRegression
