

""" Subpackage containing the BaseHunter and several subclasses which are used to determine how optimizers are
    stopped.
"""


from .basehunter import BaseHunter
from .confidencewidth import ConfidenceWidth
from .min_iterations import MinIterations
from .pseudoconv import PseudoConverged
from .val_below_asymptote import ValBelowAsymptote
from .timeannealing import TimeAnnealing
from .valueannealing import ValueAnnealing
from .parameterdistance import ParameterDistance
from .min_fcalls import MinFuncCalls
from .lastptsinvalid import LastPointsInvalid

__all__ = ("BaseHunter",
           "ConfidenceWidth",
           "MinIterations",
           "PseudoConverged",
           "ValBelowAsymptote",
           "ValueAnnealing",
           "TimeAnnealing",
           "ParameterDistance",
           "MinFuncCalls",
           "LastPointsInvalid")
