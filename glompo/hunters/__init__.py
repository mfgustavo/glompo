""" Subpackage containing the BaseHunter and several subclasses which are used to determine how optimizers are
    stopped.
"""

from .basehunter import BaseHunter
from .bestunmoving import BestUnmoving
from .evalsunmoving import EvaluationsUnmoving
from .lastptsinvalid import LastPointsInvalid
from .min_fcalls import MinFuncCalls
from .parameterdistance import ParameterDistance
from .stepsize import StepSize
from .timeannealing import TimeAnnealing
from .type import TypeHunter
from .valueannealing import ValueAnnealing

__all__ = ("BaseHunter",
           "BestUnmoving",
           "ValueAnnealing",
           "TimeAnnealing",
           "ParameterDistance",
           "MinFuncCalls",
           "LastPointsInvalid",
           "TypeHunter",
           "StepSize",
           "EvaluationsUnmoving")
