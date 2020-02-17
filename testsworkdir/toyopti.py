

import random
import time
import numpy as np
from math import log, exp


class ToyOptimizer:
    """Generates an exponential function and uses it to simulate the performance of a real optimiser."""
    def __init__(self, init: float = None, min_: float = None, steps_to_conv: int = None, noise: float = 0,
                 sec_per_eval: float = 0, elitism: bool = True, restart_chance: float = 0):
        """
        Generates the parameters and behaviours of the simulated optimizer trajectory.

        Parameters
        ----------
        init : float
            The starting value of the optimizer.
        min_ : float
            The final value of the optimizer.
        steps_to_conv : int
            The number of iterations needed to get from init to 0.99*min_.
        noise : float
            Optional parameter to perturb the function with random noise to make it more realistic.
            Function calls are multiplied by a uniformly generated random number between 1+noise and 1-noise.
        sec_per_eval : float
            Optional parameter to test dynamic properties. Sets the length of time in seconds needed for one evaluation.
        elitism : bool
            If True the trajectory will always return the lowest value evaluated thus far. If False, increases are
            allowed.
        restart_chance : float
            Fraction from 0 to 1 which describes the change that the exponential curve will regenerate, simulating a
            large dip in the optimization curve.
        """
        self._calc_params(init, min_, steps_to_conv)
        self.noise = noise
        self.elitism = elitism
        self.elite = self.init  # Tracks the elite solution
        self.iteration = 0  # Tracks the number of function evaluations
        self.sec_per_eval = sec_per_eval
        self.restart_chance = restart_chance

    def __call__(self):
        self.iteration += 1
        time.sleep(self.sec_per_eval)
        call = self._a * exp(-self._b * self.iteration) + self.min_

        if self.noise != 0:
            rand = random.uniform(1+self.noise, 1-self.noise)
            call *= rand

        if np.random.rand() < self.restart_chance:
            self.iteration = 0
            self._calc_params(init=call, min_=self.min_)

        if self.elitism:
            if call < self.elite:
                self.elite = call
            return self.elite
        else:
            return call

    def _calc_params(self, init=None, min_=None, steps_to_conv=None):
        self.init = random.gammavariate(2, 100) if init is None else init
        self.min_ = random.uniform(0, self.init) if min_ is None else min_
        self.steps_to_conv = random.randint(1, 1000) if steps_to_conv is None else steps_to_conv
        # Calculate the parameters of the exponential function
        self._a = self.init - self.min_
        self._b = (1 / self.steps_to_conv) * log(100)
