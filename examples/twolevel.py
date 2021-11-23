import logging
import sys

from glompo.benchmark_fncs import LennardJones
from glompo.convergence import MaxFuncCalls, TargetCost
from glompo.core.manager import GloMPOManager
from glompo.generators.basinhopping import BasinHoppingGenerator
from glompo.hunters import EvaluationsUnmoving
from glompo.opt_selectors import CycleSelector
from glompo.optimizers.scipy import ScipyOptimizeWrapper

try:
    from glompo.optimizers.cmawrapper import CMAOptimizer
except ModuleNotFoundError:
    raise ModuleNotFoundError("To run this example the cma package is required.")


class ShiftedLennardJones(LennardJones):
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs) + 28.422532 + 1


if __name__ == '__main__':
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s :: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger('glompo.manager')
    logger.addHandler(handler)
    logger.setLevel('INFO')

    task = ShiftedLennardJones(atoms=10, dims=3)

    max_calls = 4000  # Imposed iteration budget
    checker = MaxFuncCalls(max_calls) | TargetCost(1)  # Combined two conditions into a single stop criteria.

    generator = BasinHoppingGenerator()

    call_args = {'jac': task.jacobian}
    init_args = {'workers': 1, 'method': 'BFGS'}
    selector = CycleSelector((ScipyOptimizeWrapper, init_args, call_args))

    max_jobs = 4

    hunters = EvaluationsUnmoving(100, 0.01)

    manager = GloMPOManager.new_manager(task=task,
                                        bounds=task.bounds,
                                        opt_selector=selector,
                                        working_dir="twolevel_example_outputs",  # Dir in which files are saved
                                        overwrite_existing=True,  # Replaces existing files in working directory
                                        max_jobs=max_jobs,
                                        backend='processes',
                                        convergence_checker=checker,
                                        x0_generator=generator,
                                        killing_conditions=hunters,
                                        share_best_solutions=False,  # Send good evals from one opt to another
                                        hunt_frequency=1,  # Function evaluations between hunts
                                        status_frequency=60,  # Interval in seconds in which a status message is logged
                                        summary_files=3,  # Controls the level of output produced
                                        visualisation=True,
                                        split_printstreams=True)  # Autosend print statements from opts to files

    result = manager.start_manager()

    print("GloMPO minimum found:")
    print(result)
