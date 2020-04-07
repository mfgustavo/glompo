

""" Classes of static functions which calculate the log-posterior of the function (y0 - c) exp(-bt) + c.
    This is done through the summation of the log-likelihood and log-priors. First and second derivatives for each are
    also available for better stability of the numerical optimizations.
"""


from abc import abstractmethod
import numpy as np


__all__ = ('LogPosterior',
           'LogLikelihood')


class BaseLogProbability:
    """ Abstract class used to setup the form of the log-likelihoods and log-priors. """

    def __call__(self, *args, derivative: str = '') -> float:
        """ Returns the function or derivative value for the calculation if provided. """

        derivative_map = {'': self.calc,
                          'b': self.db,
                          'c': self.dc,
                          's': self.ds}

        method_to_call = derivative_map[derivative]

        return method_to_call(*args)

    @abstractmethod
    def calc(self, *args) -> float:
        """ Returns the log-probability. """

    def db(self, *args) -> float:
        """ Returns the first derivative with respect to b. """
        return 0

    def dc(self, *args) -> float:
        """ Returns the first derivative with respect to c. """
        return 0

    def ds(self, *args) -> float:
        """ Returns the first derivative with respect to s. """
        return 0


class LogPosterior:
    """ Calculates the log-posterior given its likelihood and prior contributions. """

    def __init__(self, c_scale: float = 0, c_shift: float = 0.01):
        """ Setups the class with a list of each contribution to the calculation.
            Accepts hyperparameter values for the combined prior on b and c (see LogPriorBC for their description).
        """
        self.contributions = [LogLikelihood(), LogPriorBC(c_scale, c_shift), LogPriorS()]

    def __call__(self, *args, derivative='') -> float:
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

    def calc(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        b, c, s = theta

        n = len(t)
        y0 = y[0]
        yn = y[-1]
        model = (y0 - c * yn) * np.exp(-b * t) + c * yn

        calc = -n * np.log(y0 * s)
        calc -= 0.5 * (y0 * s) ** -2 * np.sum((y - model) ** 2)

        if np.isnan(calc):
            return -np.inf

        return calc

    def db(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        b, c, s = theta
        y0 = y[0]
        yn = y[-1]

        exp = np.exp(b * t)
        calc = - 0.5 * (s * y0) ** -2
        calc *= np.sum(-2 * exp ** -2 * t * (y0 - c * yn) * (y0 - c * yn + exp * (c * yn - y)))

        return calc

    def dc(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        b, c, s = theta
        y0 = y[0]
        yn = y[-1]

        exp = np.exp(b * t)
        calc = - 0.5 * (s * y0) ** -2
        calc *= np.sum(2 * exp ** -2 * (exp - 1) * yn * (y0 - c * yn + exp * (c * yn - y)))

        return calc

    def ds(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        b, c, s = theta
        n = len(t)
        y0 = y[0]
        yn = y[-1]

        term1 = -n / s
        term2 = s ** -3 * y0 ** -2
        term2 *= np.sum((y - c * yn - np.exp(-b * t) * (y0 - c * yn)) ** 2)

        return term1 + term2


class LogPriorS(BaseLogProbability):
    """ s is the noise parameter accounting for deviations between the model and observed values. A Jeffery's prior (
        1/(s**2)) is selected for this parameter provided s>0.
    """

    def calc(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        s = theta[2]
        if s > 0:
            return -2 * np.log(s)
        return -np.inf

    def ds(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        s = theta[2]
        if s > 0:
            return - 2 / s
        return 0


class LogPriorBC(BaseLogProbability):
    """ Combined custom prior on B and C. This priors combines an exponential distribution on b to (bias for lower
        values) and a beta distribution on c (to swing density from high to low values depending on the value of b).
    """

    def __init__(self, c_scale: float = 0, c_shift: float = 0.01):
        """ Initialises the prior with its hyperparameters.

            Parameters
            ----------
            c_scale: float = 0
                Larger values of c_scale move density on the b and c prior to along the axes of the distribution near
                c=0 and b=0 . Defaults to zero, acceptable values are in the range [0, 10].
            c_shift: float = 0.01
                Lower values of c_shift sharpen the prior on b and c to add more density to the extremes of the
                distribution: low c / high b and high c / low b. Higher values eventually increase the probability of
                all c values at low b and removes the low c / high b peak. Defaults to 0.01, the range [0, 4].
        """
        self.c_scale = c_scale
        self.c_shift = c_shift

    def calc(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        b, c, _ = theta

        if b < 0 or c < 0 or c > 1:
            return -np.inf

        calc = -b * (self.c_scale * c + self.c_shift)
        calc += np.log(b)
        calc += (b - 1) * np.log(1 - c)
        calc += np.log(self.c_scale * c + self.c_shift)

        return calc

    def db(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        b, c, _ = theta

        calc = 1/b
        calc -= self.c_shift
        calc -= self.c_scale * c
        calc += np.log(1 - c)

        return calc

    def dc(self, theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
        b, c, _ = theta

        calc = (1 - b) / (1 - c)
        calc -= b * self.c_scale
        calc += self.c_scale / (self.c_shift + c * self.c_scale)

        return calc
