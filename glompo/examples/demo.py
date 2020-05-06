

import sys
import logging

from glompo import GloMPOManager
from glompo.optimizers import GFLSOptimizer, CMAOptimizer
from glompo.opt_selectors import CycleSelector
from glompo.convergence import KillsAfterConvergence, MaxFuncCalls
from glompo.generators import EvolutionaryStrategyGenerator
from glompo.hunters import ParameterDistance, PseudoConverged
from examples.expproblem import ExpProblem


if __name__ == '__main__':

    formatter = logging.Formatter("%(levelname)s : %(name)s : %(processName)s :: %(message)s")

    log_filter = logging.Filter('glompo')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(log_filter)
    handler.setLevel('INFO')

    logger = logging.getLogger('glompo')
    logger.addHandler(handler)
    logger.setLevel('DEBUG')

    bounds = ((-1.5, 1.5),) * 100
    manager = GloMPOManager(task=ExpProblem(0.1, 100, 50, sigma_e=0),
                            n_parms=100,
                            optimizer_selector=CycleSelector([CMAOptimizer, GFLSOptimizer]),
                            bounds=bounds,
                            working_dir="demo_output",
                            overwrite_existing=True,
                            max_jobs=4,
                            task_args=None,
                            task_kwargs=None,
                            convergence_checker=KillsAfterConvergence(2, 1) | MaxFuncCalls(10000),
                            x0_generator=EvolutionaryStrategyGenerator(bounds, 1000),
                            killing_conditions=PseudoConverged(500, 0.05) | ParameterDistance(bounds, 0.05),
                            hunt_frequency=300,
                            region_stability_check=False,
                            report_statistics=False,
                            enforce_elitism=True,
                            summary_files=3,
                            force_terminations_after=60,
                            split_printstreams=True,
                            visualisation=True,
                            visualisation_args={'record_movie': True,
                                                'x_range': (0, 10000),
                                                'y_range': (0, 15),
                                                'log_scale': True,
                                                'interactive_mode': True,
                                                'writer_kwargs': {'fps': 1},
                                                'movie_kwargs': {'outfile': 'distance_demo.mp4',
                                                                 'dpi': 200}})
    manager.start_manager()
