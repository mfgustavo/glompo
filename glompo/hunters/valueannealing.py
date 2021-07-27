import numpy as np

from .basehunter import BaseHunter
from ..core.optimizerlogger import BaseLogger

__all__ = ("ValueAnnealing",)


class ValueAnnealing(BaseHunter):
    """ Keeps optimizers alive based on the function values they are exploring.
    This condition is unlikely to kill a victim which is very near the hunter (in terms of function value) but the
    probability of killing increases with the difference between them. This condition can be applied in combination with
    others to prevent 'competitive' optimizers for being killed while still terminating poorly performing ones.

    The decision criteria follows an exponential distribution which corresponds to the probability of survival. Control
    of the probability can be achieved through the `med_kill_chance` initialisation criteria.

    Parameters
    ----------
    med_kill_chance: float
        The probability of killing a victim which is twice as large as the hunter in absolute value. The default is
        50%.

    Returns
    -------
    bool
        :obj:`True` if the difference between the explored function values of the hunter and victim fails a comparison
        test with a uniformly randomly generated number.
    """

    def __init__(self, med_kill_chance: float = 0.5):
        super().__init__()
        assert 0 < med_kill_chance < 1, "med_kill_chance must be between 0 and 1"
        self.med_kill_chance = med_kill_chance
        self.strictness = np.log(med_kill_chance)

    def __call__(self,
                 log: BaseLogger,
                 hunter_opt_id: int,
                 victim_opt_id: int) -> bool:
        f_hunter = log.get_best_iter(hunter_opt_id)['fx']
        f_victim = log.get_best_iter(victim_opt_id)['fx']

        if f_hunter == 0 or f_victim <= f_hunter:
            # Catch very unlikely corner cases
            self.last_result = False
            return self.last_result

        prob = (f_hunter - f_victim) / f_hunter
        prob = np.abs(prob)
        prob *= self.strictness
        prob = np.exp(prob)

        test_num = np.random.uniform(0, 1)

        self.last_result = test_num > prob
        return self.last_result
