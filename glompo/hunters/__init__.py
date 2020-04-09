

""" Subpackage containing the BaseHunter and several subclasses which are used to determine how optimizers are
    stopped.
"""


from .basehunter import BaseHunter
from .confidencewidth import ConfidenceWidth
from .gprsuitable import GPRSuitable
from .min_vic_trainpts import MinVictimTrainingPoints
from .pseudoconv import PseudoConverged
from .val_below_asymptote import ValBelowAsymptote
from .val_below_val import ValBelowVal
from .timeannealing import TimeAnnealing

__all__ = ("BaseHunter",
           "ConfidenceWidth",
           "GPRSuitable",
           "MinVictimTrainingPoints",
           "PseudoConverged",
           "ValBelowAsymptote",
           "ValBelowVal",
           "TimeAnnealing")
