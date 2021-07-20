from .basechecker import BaseChecker

__all__ = ("KillsAfterConvergence",)


class KillsAfterConvergence(BaseChecker):
    """ Evaluation based on the number of single optimizers converged and the number of optimizers killed thereafter.
    Returns :obj:`True` after `n_killed` optimizers have been stopped by GloMPO after `n_converged` optimizers have
    reached normal convergence.
    """

    def __init__(self, n_killed: int = 0, n_converged: int = 1):
        super().__init__()
        self.enough_conv = False
        self.kill_count = 0
        self.n_converged = n_converged
        self.n_killed = n_killed

    def __call__(self, manager: 'GloMPOManager') -> bool:
        if manager.conv_counter >= self.n_converged and not self.enough_conv:
            self.enough_conv = True
            self.kill_count = len(manager.hunt_victims)

        self.last_result = self.enough_conv and len(manager.hunt_victims) - self.kill_count >= self.n_killed
        return self.last_result
