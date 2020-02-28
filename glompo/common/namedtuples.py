

from typing import *
from multiprocessing.connection import Connection


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
