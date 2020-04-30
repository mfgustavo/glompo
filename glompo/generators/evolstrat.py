

from typing import *
from collections import deque

import numpy as np

from glompo.generators.basegenerator import BaseGenerator
from glompo.common.helpers import is_bounds_valid


class EvolutionaryStrategyGenerator(BaseGenerator):

    """ Selects new starting locations using an evolutionary algorithm approach. All iterations are fed to the
        generator. During generation, two previous iterations are randomly selected with a probability
        inversely proportional to the logarithm of their function value (i.e. better points have a higher chance of
        being selected).

        A blend crossover is used to take the weighted average of the two parent vectors using a randomly
        selected value between [-0.5, 1.5]. Each allele of the child vector is then mutated by a random value between
        [0.98, 1.02].
    """

    def __init__(self, bounds: Sequence[Tuple[float, float]], max_length: Optional[int] = None):
        """
        Parameters
        ----------
        bounds: Sequence[Tuple[float, float]]
            (min, max) bounds of each parameter.
        max_length: Optional[int]
            To reduce memory consumption as points are added to the generator only the last max_length points can be
            saved and old points are dumped as new ones are added.
        """
        super().__init__()
        if is_bounds_valid(bounds):
            self.bounds = np.array(bounds)
            self.min, self.max = tuple(np.transpose(bounds))
            self.range = np.abs(self.min - self.max)
            self.n_params = len(bounds)
        self.history = deque(maxlen=max_length)

    def generate(self) -> np.ndarray:
        if not self.history:
            return (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]

        vecs, y = np.transpose(self.history)
        y = y + 1.01 * np.abs(np.min(y)) + 1.01  # Shift values to a minimum of 1.01
        y = 1 / np.log10(y.astype(np.float64))  # log for order of magnitude differences, invert count lower nums higher
        prob_tot = np.sum(y)
        par1, par2 = np.random.choice(len(self.history), 2, p=y/prob_tot)
        par1 = np.array(vecs[par1])
        par2 = np.array(vecs[par2])

        # Blend Crossover
        alpha = np.random.uniform(-0.5, 1.5)
        child = alpha * par1 + (1-alpha) * par2

        # Mutation
        mut_rate = 0.02
        child *= np.random.uniform(1-mut_rate, 1+mut_rate, self.n_params)

        # Clip back into (but not onto) bounds
        child = np.clip(child, self.min + self.range*0.0001, self.max - self.range*0.0001)

        return child

    def update(self, x: Sequence[float], fx: float):
        self.history.append((x, fx))
