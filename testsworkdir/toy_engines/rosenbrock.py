

from time import sleep


class Rosenbrock:
    """ When called returns evaluations of the Rosenbrock function. """

    def __init__(self, dims: int, delay: int = 0):
        """
        Implementation of the Rosenbrock optimization test function.

        Parameters
        ----------
        dims : int
            Number of dimensions of the function.
        delay : int
            Delay in seconds after the function is called before results are returned.
            Critical to simulating harder problems and testing GloMPO management.
        """
        self.dims = dims
        self.delay = delay

    def __call__(self, x):
        total = 0
        for i in range(self.dims-1):
            total += 100 * (x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        sleep(self.delay)
        return total
