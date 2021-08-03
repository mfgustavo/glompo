import numpy as np

from glompo.core.manager import GloMPOManager
from .basegenerator import BaseGenerator

__all__ = ("BasinHoppingGenerator",)


class BasinHoppingGenerator(BaseGenerator):
    """ Monte-Carlo sampling strategy used by the Basin-Hopping algorithm.
    Represents the 'outer' algorithm used by basin-hopping to select locations at which to start local optimizers.

    Parameters
    ----------
    temperature
        Parameter for the accept or reject criterion. Higher values mean larger jumps will be accepted from the
        generator's starting point.
    max_step_size
        Maximum jump size allowed.
    target_accept_rate
        The target acceptance rate. The rate is calculated as the ratio between the number of optimizers which found a
        better minimum than the manager's incumbent, and the number of optimizers started.
    interval
        The number of :meth:`~.BaseGenerator.generate` calls between adjustments of the step size based on the
        acceptance rate.
    factor
        The factor by which the step size is adjusted when an adjustment is done.

    Notes
    -----
    The generator attempts to closely follow the basin-hopping sampling strategy, however, due to GloMPO's inherent
    parallelism, several adjustments are made. The generation algorithm works as follows:

       #. Calls to :meth:`~.BaseGenerator.generate` will return a random vector if the manager does not yet have an
          incumbent solution.

       #. If an incumbent exists, the generator's 'location' will be placed there.

       #. From the other children, one is uniformly randomly selected, and its best solution selected. There is a Monte-
          Carlo chance that the generator's 'location' will be moved to this location. In this way some diversity is
          maintained and an optimizer may be started in a different region.

       #. If the new optimizer is a multiple of interval, than the step size is grown or shrunk based on if the realised
          acceptance rate is above or below the target acceptance rate.

       #. A vector is returned which adds or subtracts a uniform random value between zero and step size to each element
          of the generator's 'location'.

    See Also
    --------
    :func:`scipy.optimize.basinhopping`

    References
    ----------
    Adapted from: SciPY basin-hopping algorithm implementation.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
    """

    def __init__(self,
                 temperature: float = 1,
                 max_step_size: float = 0.5,
                 target_accept_rate=0.5,
                 interval=5,
                 factor=0.9):
        super().__init__()
        self.beta = 1.0 / temperature if temperature != 0 else float('inf')
        self.n_accept = -1  # Counts the number of times a point has improved or is the same when generate is called.
        self.state = {'opt_id': 0,
                      'fx': float('inf')}

        # Step size related attributes
        self.max_step_size = self.step_size = max_step_size
        self.target_accept_rate = target_accept_rate
        self.interval = interval
        self.factor = factor

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        best = manager.opt_log.get_best_iter()

        if best['fx'] == float('inf'):
            # No good iteration found yet, return random start location
            self.logger.debug("Iteration history not found, returning random start location.")
            bnds = np.array(manager.bounds).T
            return np.random.uniform(bnds[0], bnds[1], manager.n_parms)

        self.logger.debug("Best value found at %f from optimizer %d", best['fx'], best['opt_id'])
        if best['fx'] <= self.state['fx'] and best['opt_id'] != self.state['opt_id']:
            # Another optimizer found the same or better minimum than the one known.
            self.n_accept += 1
            self.state = {'opt_id': best['opt_id'], 'fx': best['fx']}
            self.logger.debug("Improvement detected. Number of accepted points: %d", self.n_accept)

        # Metropolis chance of accepting another optimizer that is not the best
        avail_keys = set(manager.opt_log._best_iters.keys())
        avail_keys.remove(0)
        avail_keys.remove(manager.opt_log.get_best_iter()['opt_id'])
        if avail_keys:
            other = manager.opt_log._best_iters[np.random.choice([*avail_keys])]
            w = np.exp(np.min([0, -(other['fx'] - best['fx']) * self.beta]))
            rand = np.random.random()
            if w >= rand:
                self.logger.debug("Opt %d (%f) replacing best.", other['opt_id'], other['fx'])
                best = other
                self.n_accept += 1

        # Adjust step size
        if manager.o_counter % self.interval == 0:
            self.logger.debug("Step size being adjusted.")
            old_step = self.step_size
            actual_accept_rate = self.n_accept / manager.o_counter
            self.logger.debug("Accept rate at %f", actual_accept_rate)
            if actual_accept_rate > self.target_accept_rate:
                self.step_size /= self.factor
                self.logger.debug("Accept rate too high, step size increasing %f -> %f.", old_step, self.step_size)
            elif actual_accept_rate < self.target_accept_rate:
                self.step_size *= self.factor
                self.logger.debug("Accept rate too low, step size decreasing %f -> %f.", old_step, self.step_size)

        # Random perturbation of the best vector based on the step size
        x = best['x'].copy()
        x += np.random.uniform(-self.step_size, self.step_size, x.size)

        return x
