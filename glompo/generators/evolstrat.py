

from typing import *

import numpy as np

from glompo.generators.basegenerator import BaseGenerator
from glompo.common.helpers import is_bounds_valid


class EvolutionaryStrategyGenerator(BaseGenerator):

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        if is_bounds_valid(bounds):
            self.bounds = np.array(bounds)
            self.min, self.max = tuple(np.transpose(bounds))
            self.n_params = len(bounds)
        self.history = {}

    def generate(self) -> np.ndarray:
        if not self.history:
            return (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.random(self.n_params) + self.bounds[:, 0]

        prob_vec = 1/np.array([*self.history.values()])
        prob_tot = np.sum(prob_vec)
        par1, par2 = np.random.choice(len([*self.history]), 2, p=prob_vec/prob_tot)
        par1 = [*self.history][par1]
        par2 = [*self.history][par2]

        # Blend Crossover
        alpha = np.random.uniform(-0.5, 1.5, self.n_params)
        child = alpha * par1 + (1-alpha) * par2

        # Mutation
        mut_rate = 0.02
        child *= np.random.uniform(1-mut_rate, 1+mut_rate, self.n_params)

        # Clip back into bounds
        child = np.clip(child, self.min, self.max)

        return child

    def update(self, x: Sequence[float], fx: float):
        self.history[tuple(x)] = fx
