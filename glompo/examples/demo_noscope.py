

import sys
import logging

from glompo import GloMPOManager
from glompo.optimizers.gflswrapper import GFLSOptimizer
from glompo.optimizers.cmawrapper import CMAOptimizer
from glompo.opt_selectors import CycleSelector
from glompo.convergence import KillsAfterConvergence, MaxFuncCalls
from glompo.generators import ExploitExploreGenerator
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
                            optimizer_selector=CycleSelector([CMAOptimizer, GFLSOptimizer]),
                            bounds=bounds,
                            working_dir="demo_output",
                            overwrite_existing=True,
                            max_jobs=4,
                            task_args=None,
                            task_kwargs=None,
                            convergence_checker=KillsAfterConvergence(2, 1) | MaxFuncCalls(10000),
                            x0_generator=ExploitExploreGenerator(bounds, 10000),
                            killing_conditions=PseudoConverged(500, 0.05) | ParameterDistance(bounds, 0.05),
                            hunt_frequency=300,
                            region_stability_check=False,
                            report_statistics=False,
                            enforce_elitism=True,
                            summary_files=3,
                            force_terminations_after=60,
                            split_printstreams=True,
                            visualisation=False)
    manager.start_manager()
