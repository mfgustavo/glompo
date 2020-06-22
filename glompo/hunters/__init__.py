

""" Subpackage containing the BaseHunter and several subclasses which are used to determine how optimizers are
    stopped.
"""


from .basehunter import BaseHunter
from .min_iterations import MinIterations
from .pseudoconv import PseudoConverged
from .timeannealing import TimeAnnealing
from .valueannealing import ValueAnnealing
from .parameterdistance import ParameterDistance
from .min_fcalls import MinFuncCalls
from .lastptsinvalid import LastPointsInvalid
from .type import TypeHunter

__all__ = ("BaseHunter",
           "MinIterations",
           "PseudoConverged",
           "ValueAnnealing",
           "TimeAnnealing",
           "ParameterDistance",
           "MinFuncCalls",
           "LastPointsInvalid",
           "TypeHunter")
