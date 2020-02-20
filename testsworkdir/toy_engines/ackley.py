

from time import sleep
import numpy as np


class Ackley:
    """ When called returns evaluations of the Ackley function. """

    def __init__(self, delay: int = 0):
        """
        Implementation of the Ackley optimization test function.

        Parameters
        ----------
        delay : int
            Delay in seconds after the function is called before results are returned.
            Critical to simulating harder problems and testing GloMPO management.
        """
        self.delay = delay

    def __call__(self, x, y):
        calc = np.exp(1) + 20
        calc -= np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
        calc -= 20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        sleep(self.delay)
        return calc
