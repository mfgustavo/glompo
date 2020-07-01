

""" Subpackage containing the BaseHunter and several subclasses which are used to determine how optimizers are
    stopped.
"""


from .basehunter import BaseHunter
from .lastptsinvalid import LastPointsInvalid
from .min_fcalls import MinFuncCalls
from .min_iterations import MinIterations
from .parameterdistance import ParameterDistance
from .pseudoconv import PseudoConverged
from .timeannealing import TimeAnnealing
from .type import TypeHunter
from .valueannealing import ValueAnnealing

__all__ = ("BaseHunter",
           "MinIterations",
           "PseudoConverged",
           "ValueAnnealing",
           "TimeAnnealing",
           "ParameterDistance",
           "MinFuncCalls",
           "LastPointsInvalid",
           "TypeHunter")
