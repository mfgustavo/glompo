from typing import Iterable, Sequence, Tuple, Union

import numpy as np
from scipy.optimize._dual_annealing import EnergyState, ObjectiveFunWrapper, StrategyChain, VisitingDistribution

from .basegenerator import BaseGenerator

__all__ = ("AnnealingGenerator",)


class AnnealingGenerator(BaseGenerator):
    """ Wrapper around the core of :func:`scipy.optimize.dual_annealing`.

    The algorithm is adapted directly from Scipy and directly uses its internal code. Each generation performs several
    function evaluations to select a location from which to start a new optimizer. The dual-annealing methodology is
    followed as closely as possible but given GloMPO's parallel and asynchronous behaviour, some adjustments are needed.

    For each generation:

    #. The 'state' of the optimizer is updated to the best seen thus far by any of the manager's children.

    #. The temperature is decreased. If it reaches a critically low level it is reset back to the initial temperature.

    #. Run the internal annealing chain. If the 'state' is different from the start location of the previous optimizer,
       it is returned.

    #. Otherwise, the annealing chain is repeated (a maximum of 5 times). If a new location is still not found, the
       temperature is reset and the procedure returns to Step 3.

    .. danger::

       This generator performs function evaluations everytime :meth:`~.BaseGenerator.generate` is
       called! This is not the typical GloMPO design intention. If one is using slow functions, this could significantly
       impede the manager!

    Parameters
    ----------
    bounds
        Sequence of (min, max) pairs for each parameter in the search space.
    task
        The optimization function.
    qa
        The accept distribution parameter.
    qv
        The visiting distribution parameter.
    initial_temp
        Initial temperature. Larger values generate larger step sizes.
    restart_temp_ratio
        The value of :code:`temperature / initial_temp` at which the temperature is reset to the initial value.
    seed
        Seed for the random number generator for reproducibility.
    """

    def __init__(self,
                 bounds: Sequence[Tuple[float, float]],
                 task,
                 qa: float = -5.0,
                 qv: float = 2.62,
                 initial_temp: float = 5230,
                 restart_temp_ratio: float = 2e-5,
                 seed: Union[None, int, np.ndarray, Iterable, float] = None):
        super().__init__()

        self.lb, self.ub = np.array(bounds).T
        self.rand_state = np.random.RandomState(seed)

        self.n_params = len(bounds)
        self.current_fx = float('inf')
        self.current_x = []
        self.iter = -1
        self.last = np.empty(self.n_params)

        self.initial_temperature = initial_temp
        self.temperature = initial_temp
        self.restart_temperature = initial_temp * restart_temp_ratio
        self.qv = qv
        self.qa = qa
        self.t1 = np.exp((self.qv - 1) * np.log(2)) - 1

        # Scipy Internals
        self.obj_wrapper = ObjectiveFunWrapper(task)
        self.visiting_dist = VisitingDistribution(self.lb, self.ub, qv, self.rand_state)
        self.state = EnergyState(self.lb, self.ub)
        self.state.reset(self.obj_wrapper, self.rand_state)
        self.chain = StrategyChain(qa, self.visiting_dist, self.obj_wrapper, None, self.rand_state, self.state)

    def generate(self, manager: 'GloMPOManager') -> np.ndarray:
        self.iter += 1

        # Update with results from children
        best = manager.opt_log.get_best_iter()
        if best['fx'] < self.state.current_energy:
            self.logger.debug("State updated from children.")
            self.state.update_best(best['fx'], best['x'].copy(), None)
            self.state.update_current(best['fx'], best['x'].copy())

        reps = 0
        while True:
            reps += 1

            # Update temperature
            t2 = np.exp((self.qv - 1) * np.log(self.iter + 2)) - 1
            self.temperature = self.initial_temperature * self.t1 / t2
            self.logger.debug("Temperature updated to %f", self.temperature)
            if self.temperature < self.restart_temperature:
                self.logger.debug("Temperature below restart temperature, resetting.")
                self.reset_temperature()

            # Run the internal annealing chain and update the number of function evaluations used to the manager
            # DANGER ZONE! Adjusting manager properties can have unforeseen consequences!
            evals_0 = self.obj_wrapper.nfev
            self.chain.run(self.iter, self.temperature)
            manager.f_counter += self.obj_wrapper.nfev - evals_0
            manager.opt_log._f_counter += self.obj_wrapper.nfev - evals_0

            if np.all(np.isclose(self.state.current_location, self.last)):
                self.logger.debug("Current location too close to previous one, reannealing.")
                if reps >= 5:
                    self.logger.debug("To many reannealings without a location update, resetting temperature.")
                    self.reset_temperature()
                    reps = 0
            else:
                break

        self.last = self.state.current_location.copy()
        return self.state.current_location

    def reset_temperature(self):
        self.state.reset(self.obj_wrapper, self.rand_state)
        self.iter = 0
        self.temperature = self.initial_temperature
