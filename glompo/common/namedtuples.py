

from typing import *


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


class HuntingResult(NamedTuple):
    hunt_id: int
    victim: int


class IterationResult(NamedTuple):
    opt_id: int
    n_iter: int
    x: Sequence[float]
    fx: float
    final: bool  # True if this is the final result sent to the queue by this optimizer


class HyperparameterOptResult(NamedTuple):
    opt_id: int
    alpha: float
    beta: float
    sigma: float
