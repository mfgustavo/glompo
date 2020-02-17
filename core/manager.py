

# Native Python imports
from typing import *
import multiprocessing as mp

# Package imports
from scope.scope import ParallelOptimizerScope
from optimizers.baseoptimizer import BaseOptimizer


class GloMPOOptimizer:
    """ Runs given jobs in parallel and tracks their progress using Gaussian Process Regressions.
        Based on these predictions the class will update hyperparameters, kill poor performing jobs and
        intelligently restart others. """

    def __init__(self,
                 optimizers: dict,
                 max_jobs: int,
                 visualisation: bool = False,
                 visualisation_args: Union[dict, None] = None):
        """
        Generates the environment for a globally managed parallel optimization job.

        Parameters
        ----------
        optimizers: dict, Callables
            Dictionary of callable optimization functions with keywords describing their behaviour. Recognized keywords
            are:
                'default': The default optimizer used if any of the below keywords are not set. This is the only
                    optimizer which *must* be set.
                'early': Optimizers which are more global in their search and best suited for early stage optimization.
                'late': Strong local optimizers which are best suited to refining solutions in regions with known good
                    solutions.
                'noisy': Optimizer which should be used in very noisy areas with very steep gradients or discontinuities

        max_jobs : int
            The maximum number of local optimizers run in parallel.
        visualisation : bool
            If True then a dynamic plot is generated to demonstrate the performance of the optimizers. Further options
            (see visualisation_args) allow this plotting to be recorded and saved as a film.
        visualisation_args : Union[dict, None]
            Optional arguments to parameterize the dynamic plotting feature.
        """

        # TODO Also implement check to see if it is an instance of an approved base class
        if 'default' not in optimizers:
            raise ValueError("'default' not found in optimizer dictionary. This value must be set.")
        for optimizer in optimizers.values():
            if not isinstance(optimizer, BaseOptimizer):
                raise TypeError(f"{optimizer} not an instance of BaseOptimizer.")

        self.optimizers = optimizers

        self.max_jobs = max_jobs

        self.optimizer_jobs = []
        self.hyperparm_jobs = []
        self.hunting_jobs = []

        self.manager = mp.Manager()
        self.optimizer_queue = self.manager.Queue()
        self.hyperparm_queue = self.manager.Queue()
        self.hunting_queue = self.manager.Queue()

        if visualisation:
            self.scope = ParallelOptimizerScope(num_streams=max_jobs, **visualisation_args)

    def start_manager(self):
        """ Begins the optimization routine. """
        pass

    # TODO Selection of new starting points can be driven by a Gaussian Process (Probably unrealistic unless a LOT of
    #  optimizers are started), Genetic Algorithms (Reasonable but still no feeling for the error surface) or random
    #  (very easy to implement and possible same quality as the others given the sparse space).
    def _start_new_job(self):
        pass

    def _kill_job(self):
        pass

    def _start_hunt(self):
        pass

    def _optimize_hyperparameters(self):
        pass

    def _explore_basin(self):
        pass
