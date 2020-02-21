

from time import sleep


class DataDrawer:
    """ Initialised with a data file of real optimization history. When called delivers the next element in the
    sequence. """

    def __init__(self, file: str, delay: int = 0):
        """
        Implementation of the Rosenbrock optimization test function.

        Parameters
        ----------
        file : str
            Path to and name of file from which data is extracted. File is expected to have the form: i fx
            with the first line being a header.
        delay : int
            Delay in seconds after the function is called before results are returned.
            Critical to simulating harder problems and testing GloMPO management.
        """
        with open(file, 'r') as f:
            lines = f.readlines()
            self.data = []
            for line in lines[1:]:
                self.data.append(float(line.split(' ')[1]))
        self.delay = delay
        self.iter = -1

    def __call__(self, x):
        self.iter = self.iter + 1 if self.iter < len(self.data) - 1 else self.iter
        return self.data[self.iter]
