

""" Subpackage containing the BaseChecker and several subclasses which are used to determine GloMPO convergence. """

from .basechecker import BaseChecker

from .nkillsafterconv import KillsAfterConvergence
from .fmax import MaxFuncCalls
from .tmax import MaxSeconds
from .omax import MaxOptsStarted
from .nkills import MaxKills
from .nconv import NOptConverged

__all__ = ("BaseChecker",
           "KillsAfterConvergence",
           "MaxFuncCalls",
           "MaxSeconds",
           "MaxKills",
           "MaxOptsStarted",
           "NOptConverged")
