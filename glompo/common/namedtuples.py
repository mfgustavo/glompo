

from typing import *
from multiprocessing import Process, Event
from multiprocessing.connection import Connection
from ..core.gpr import GaussianProcessRegression


__all__ = ("Result",
           "Bound",
           "OptimizerPackage",
           "HuntingResult",
           "IterationResult",
           "HyperparameterOptResult",
           "ProcessPackage")


class Result(NamedTuple):
    x: Sequence[float]
    fx: float
    stats: Dict[str, Any]
    origin: Dict[str, Any]  # Optimizer name, settings, starting point and termination condition


class Bound(NamedTuple):
    min: float
    max: float


class OptimizerPackage(NamedTuple):
    opt_id: int
    optimizer: Callable
    call_kwargs: Dict[str, Any]
    signal_pipe: Connection
    allow_run_event: Event
    gpr: GaussianProcessRegression


class HuntingResult(NamedTuple):
    hunt_id: int
    victims: Sequence[int]


class IterationResult(NamedTuple):
    opt_id: int
    n_iter: int
    n_icalls: int  # Number of functions calls in iteration *not* cumulative calls
    x: Sequence[float]
    fx: float
    final: bool  # True if this is the final result sent to the queue by this optimizer


class HyperparameterOptResult(NamedTuple):
    hyper_id: int
    opt_id: int
    alpha: float
    beta: float
    sigma: float


class ProcessPackage(NamedTuple):
    process: Process
    signal_pipe: Connection
    allow_run_event: Event
    gpr: GaussianProcessRegression
