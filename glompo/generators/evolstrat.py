

from typing import *

import numpy as np

from glompo.generators.basegenerator import BaseGenerator
from glompo.common.helpers import is_bounds_valid


class EvolutionaryStrategyGenerator(BaseGenerator):

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        super().__init__()
        if is_bounds_valid(bounds):
            self.bounds = np.array(bounds)
            self.min, self.max = tuple(np.transpose(bounds))
            self.range = np.abs(self.min - self.max)
            self.n_params = len(bounds)
        self.history = {}

    def generate(self) -> np.ndarray:
        if not self.history:
            return (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]

        x = [*self.history.values()]
        y = x + 1.01 * np.abs(np.min(x)) + 1.01  # Shift values to a minimum of 1.01
        y = 1 / np.log10(y)  # Take the log for order of magnitude differences and invert to weigh lower numbers higher
        prob_tot = np.sum(y)
        par1, par2 = np.random.choice(len([*self.history]), 2, p=y/prob_tot)
        par1 = [*self.history][par1]
        par2 = [*self.history][par2]

        # Blend Crossover
        alpha = np.random.uniform(-0.5, 1.5, self.n_params)
        child = alpha * par1 + (1-alpha) * par2

        # Mutation
        mut_rate = 0.02
        child *= np.random.uniform(1-mut_rate, 1+mut_rate, self.n_params)

        # Clip back into (but not onto) bounds
        child = np.clip(child, self.min + self.range*0.0001, self.max - self.range*0.0001)

        return child

    def update(self, x: Sequence[float], fx: float):
        self.history[tuple(x)] = fx
