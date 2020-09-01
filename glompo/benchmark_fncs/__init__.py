""" Collection of benchmark functions which are helpful for testing purposes. Can be used to experiment with different
    configurations and ensure a script is functional before being applied to a more expensive test case.
"""

from ._base import BaseTestCase
from .ackley import Ackley
from .alpine01 import Alpine01
from .alpine02 import Alpine02
from .deceptive import Deceptive
from .easom import Easom
from .griewank import Griewank
from .langermann import Langermann
from .leastsqrs import ExpLeastSquaresCost
from .levy import Levy
from .michalewicz import Michalewicz
from .qing import Qing
from .rana import Rana
from .rastrigin import Rastrigin
from .rosenbrock import Rosenbrock
from .schwefel import Schwefel
from .shekel import Shekel
from .shubert import Shubert
from .stochastic import Stochastic
from .styblinskitang import StyblinskiTang
from .trigonometric02 import Trigonometric
from .vincent import Vincent
from .zerosum import ZeroSum

__all__ = ("BaseTestCase",
           "Ackley",
           "Deceptive",
           "Easom",
           "Langermann",
           "Levy",
           "Michalewicz",
           "Rastrigin",
           "Rosenbrock",
           "Schwefel",
           "Shekel",
           "Shubert",
           "Trigonometric",
           "Vincent",
           "ZeroSum",
           "Alpine01",
           "Alpine02",
           "Griewank",
           "ExpLeastSquaresCost",
           "Qing",
           "Rana",
           "Stochastic",
           "StyblinskiTang")
