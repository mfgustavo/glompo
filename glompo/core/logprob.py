

""" Classes of static functions which calculate the log-posterior of the function (y0 - c) exp(-bt) + c.
    This is done through the summation of the log-likelihood and log-priors. First and second derivatives for each are
    also available for better stability of the numerical optimizations.
"""


from typing import *
from abc import abstractmethod
import numpy as np


__all__ = ('LogPosterior',
           'LogLikelihood')


class BaseLogProbability:
    """ Abstract class used to setup the form of the log-likelihoods and log-priors. """

    def __call__(self, *args, derivative=''):
        """ Returns the function or derivative value for the calculation if provided. """

        derivative_map = {'': self.calc,
                          'b': self.db,
                          'c': self.dc,
                          's': self.ds}

        method_to_call = derivative_map[derivative]

        return method_to_call(*args)

    @staticmethod
    @abstractmethod
    def calc(*args):
        """ Returns the log-probability. """

    @staticmethod
    def db(*args):
        """ Returns the first derivative with respect to b. """
        return 0

    @staticmethod
    def dc(*args):
        """ Returns the first derivative with respect to c. """
        return 0

    @staticmethod
    def ds(*args):
        """ Returns the first derivative with respect to s. """
        return 0


class LogPosterior:
    """ Calculates the log-posterior given its likelihood and prior contributions. """

    def __init__(self):
        """ Setups the class with a list of each contribution to the calculation. """
        self.contributions = [LogLikelihood(), LogPriorB(), LogPriorC(), LogPriorS()]

    def __call__(self, *args, derivative=''):
        """ Returns the function or derivative value for the calculation if provided. """

        total = 0
        for cont in self.contributions:
            total += cont(*args, derivative=derivative)

        return total


class LogLikelihood(BaseLogProbability):
    """ The log-likelihood is the log of a normal distribution around the model value of each value y as given in
        (Bayesian and Frequentist Regression Methods, Wakefield, 2013). The scale of the normal distribution is given
        by y0 * s for ease of interpretation since s becomes a unitless measure.
    """

    @staticmethod
    def calc(theta, t, y):
        b, c, s = theta

        n = len(t)
        y0 = y[0]
        model = (y0 - c) * np.exp(-b * t) + c

        calc = -n * np.log(y0 * s)
        calc -= 0.5 / ((y0 * s) ** 2) * np.sum((y - model) ** 2)

        if np.isnan(calc):
            return -np.inf

        return calc

    @staticmethod
    def db(theta, t, y):
        b, c, s = theta
        y0 = y[0]

        exp = np.exp(b * t)
        calc = - 0.5 * (s * y0) ** -2
        calc *= np.sum(2 * exp ** -2 * t * (c - y0) * (y0 - c + exp * (c - y)))

        return calc

    @staticmethod
    def dc(theta, t, y):
        b, c, s = theta
        y0 = y[0]

        exp = np.exp(b * t)
        calc = - 0.5 * (s * y0) ** -2
        calc *= np.sum(2 * exp ** -2 * (exp - 1) * (y0 - c + exp * (c - y)))

        return calc

    @staticmethod
    def ds(theta, t, y):
        b, c, s = theta
        n = len(t)
        y0 = y[0]

        term1 = -n / s
        term2 = s ** -3 * y0 ** -2
        term2 *= np.sum((y - c - np.exp(-b * t) * (y0 - c)) ** 2)

        return term1 + term2


class LogPriorB(BaseLogProbability):
    """ b is the rate parameter of the exponential decay. An exponential prior is selected for this parameter. This
        enforces that only positive values are allowed (the exponential will only decay not increase). A scale value
        of 2 is chosen for this prior, this puts most of the mass density at lower values of b below 1 since values
        larger than 1 are extremely steep and become difficult to differentiate.
    """

    @staticmethod
    def calc(theta, *args):
        b = theta[0]
        if b > 0:
            return -2 * b
        return -np.inf

    @staticmethod
    def db(theta, *args):
        b = theta[0]
        if b > 0:
            return -2
        return 0


class LogPriorC(BaseLogProbability):
    """ c is the value of the asymptote to which the exponential decays. A normal distribution centered at zero with
        a standard deviation of y0 is placed on this parameter.
    """

    @staticmethod
    def calc(theta, t, y):
        c = theta[1]
        y0 = y[0]
        return -0.5 * (c / y0) ** 2

    @staticmethod
    def dc(theta, t, y):
        c = theta[1]
        y0 = y[0]
        return - y0 ** -2 * c


class LogPriorS(BaseLogProbability):
    """ s is the noise parameter accounting for deviations between the model and observed values. A Jeffery's prior (
        1/(s**2)) is selected for this parameter provided s>0.
    """

    @staticmethod
    def calc(theta, t, y):
        s = theta[2]
        if s > 0:
            return -2 * np.log(s)
        return -np.inf

    @staticmethod
    def ds(theta, t, y):
        s = theta[2]
        if s > 0:
            return - 2 / s
        return 0
