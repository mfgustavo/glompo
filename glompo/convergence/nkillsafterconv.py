

from .basechecker import BaseChecker


class KillsAfterConvergence(BaseChecker):
    """ This class is used to determine GloMPO convergence based on the number of single optimizers converged and the
        number of optimizers killed thereafter.
    """

    def __init__(self, n_killed: int = 0, n_converged: int = 1):
        """ Convergence is reached after n_killed optimizers have been stopped by GloMPO after n_converged optimizers
            have reached normal convergence.
        """
        self.enough_conv = False
        self.kill_count = 0
        self.n_converged = n_converged
        self.n_killed = n_killed

    def converged(self, manager: 'GloMPOManager') -> bool:
        if manager.conv_counter >= self.n_converged:
            self.enough_conv = True
            self.kill_count = manager.kill_counter

        if self.enough_conv and manager.kill_counter - self.kill_count >= self.n_killed:
            return True
        else:
            return False