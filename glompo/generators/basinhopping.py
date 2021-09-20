import numpy as np
from glompo.core.manager import GloMPOManager

from .basegenerator import BaseGenerator

__all__ = ("BasinHoppingGenerator",)


# TODO Document
# TODO Adapted from scipy implementation
class BasinHoppingGenerator(BaseGenerator):
    def __init__(self,
                 temperature: float = 1,
                 step_size: float = 0.5,
                 accept_rate=0.5,
                 interval=5,
                 factor=0.9):
        super().__init__()
        self.beta = 1.0 / temperature if temperature != 0 else float('inf')
        self.n_accept = 0

        # Step size related attributes
        self.step_size = step_size
        self.accept_rate = accept_rate
        self.interval = interval
        self.factor = factor

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        """
        Best location is defined by the best seen thus far in the optimization. This vector is perturbed but there is a
        Metropolis chance of selecting another starting point.
        """
        best = manager.opt_log.get_best_iter()

        if best['fx'] == float('inf'):
            # No good iteration found yet, return random start location
            self.logger.debug("Iteration history not found, returning random start location.")
            bnds = np.array(manager.bounds).T
            return np.random.uniform(bnds[0], bnds[1], manager.n_parms)

        # Metropolis chance of accepting another optimizer that is not the best
        # If several optimizers return True, the first is accepted
        self.logger.debug("Best value found at %f from optimizer %d", best['fx'], best['opt_id'])
        for k, v in manager.opt_log._best_iters.items():
            if k == best['opt_id']:
                continue

            w = np.exp(np.min([0, -(v['fx'] - best['fx']) * self.beta]))
            rand = np.random.random()
            if w >= rand:
                self.logger.debug("Opt %d (%f) replacing best.", v['opt_id'], v['fx'])
                best = v
                self.n_accept += 1
                break

        # Adjust step size
        if manager.o_counter % self.interval == 0:
            self.logger.debug("Step size being adjusted.")
            old_step = self.step_size
            actual_accept_rate = self.n_accept / manager.o_counter
            self.logger.debug("Accept rate at %f", actual_accept_rate)
            if actual_accept_rate > self.accept_rate:
                self.step_size /= self.factor
                self.logger.debug("Accept rate too high, step size %f -> %f.", old_step, self.step_size)
            elif actual_accept_rate < self.accept_rate:
                self.step_size *= self.factor
                self.logger.debug("Accept rate too low, step size %f -> %f.", old_step, self.step_size)

        # Random perturbation of the best vector based on the step size
        x = best['x'].copy()
        x += np.random.uniform(-self.step_size, self.step_size, x.size)

        assert np.all(x != best['x'])

        return x
