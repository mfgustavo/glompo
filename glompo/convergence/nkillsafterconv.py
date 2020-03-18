

from .basechecker import BaseChecker


__all__ = ("KillsAfterConvergence",)


class KillsAfterConvergence(BaseChecker):
    """ This class is used to determine GloMPO convergence based on the number of single optimizers _converged and the
        number of optimizers killed thereafter.
    """

    def __init__(self, n_killed: int = 0, n_converged: int = 1):
        """ Convergence is reached after n_killed optimizers have been stopped by GloMPO after n_converged optimizers
            have reached normal convergence.
        """
        super().__init__()
        self.enough_conv = False
        self.kill_count = 0
        self.n_converged = n_converged
        self.n_killed = n_killed

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        if manager.conv_counter >= self.n_converged:
            self.enough_conv = True
            self.kill_count = len(manager.hunt_victims)

        self._converged = self.enough_conv and len(manager.hunt_victims) - self.kill_count >= self.n_killed
        return self._converged
