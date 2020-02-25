

import numpy as np


class _SEKernel:
    """ Implements and calculates instances of the squared-exponential covariance function. """

    @staticmethod
    def _norm(x: np.ndarray) -> float:
        return np.sqrt(np.sum(x ** 2))

    def __init__(self, len_scale: float = 1, sigma_signal: float = 1, sigma_noise: float = 0):
        """ Initialises the kernel hyper-parameters.

            Parameters:
            -----------
            len_scale : float
                Length scale hyper-parameter.
            sigma_signal : float
                Standard deviation of the signal.
            sigma_noise : float
                Standard deviation of the noise in given data points.
        """
        self.len_scale = len_scale
        self.sigma_signal = sigma_signal
        self.sigma_noise = sigma_noise

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        calc = self.sigma_signal ** 2
        calc *= np.exp(- 0.5 * self.len_scale ** -2 * self._norm(x1 - x2) ** 2)
        calc += self.sigma_noise ** 2 * np.all(x1 == x2)
        return calc
