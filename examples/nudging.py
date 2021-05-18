import logging
import sys

from glompo.benchmark_fncs import Schwefel
from glompo.convergence import MaxFuncCalls, TargetCost
from glompo.core.checkpointing import CheckpointingControl
from glompo.core.manager import GloMPOManager
from glompo.generators import RandomGenerator
from glompo.hunters import BestUnmoving, EvaluationsUnmoving
from glompo.opt_selectors import CycleSelector

try:
    from glompo.optimizers.cmawrapper import CMAOptimizer
except ModuleNotFoundError:
    raise ModuleNotFoundError("To run this example the cma package is required.")

if __name__ == '__main__':
    """ This example is a variation of the one in 'customized.py'. GloMPO will be run on the same task with virtually 
        the same configuration, but in this case good iteration will be shared between optimizers. The optimizers, 
        in turn, will use this information to accelerate their convergence.
    """
    task = Schwefel(dims=20)
    max_calls = 100000
    checker = MaxFuncCalls(max_calls) | TargetCost(task.min_fx)

    sigma = (task.bounds[0][1] - task.bounds[0][0]) / 2
    call_args = {'sigma0': sigma}
    """ In this case we tell CMA to accept suggestions from the manager and sample these points once every 10 
        iterations.
    """
    init_args = {'workers': 1, 'popsize': 12, 'force_injects': True, 'injection_frequency': 10}

    selector = CycleSelector((CMAOptimizer, init_args, call_args))
    max_jobs = 10
    """ The hunting must be reconfigured slightly to better suit the new optimization behavior. """
    hunters = EvaluationsUnmoving(100, 0.01) | BestUnmoving(int(max_calls / 15), 0.2)
    generator = RandomGenerator(task.bounds)

    backend = 'processes'

    visualisation = True
    visualisation_args = {'record_movie': True,
                          'x_range': (0, max_calls),
                          'y_range': None,
                          'log_scale': False,
                          'events_per_flush': 500,
                          'interactive_mode': True,
                          'writer_kwargs': {'fps': 8},
                          'movie_kwargs': {'outfile': 'distance_demo.mp4',
                                           'dpi': 200}}

    force_terminations = -1

    checkpointing = CheckpointingControl(checkpoint_at_conv=True,
                                         naming_format='nudging_completed_%(date)_%(time).tar.gz',
                                         checkpointing_dir="nudging_example_outputs")

    manager = GloMPOManager.new_manager(task=task,
                                        bounds=task.bounds,
                                        opt_selector=selector,
                                        working_dir="nudging_example_outputs",
                                        overwrite_existing=True,
                                        max_jobs=max_jobs,
                                        backend=backend,
                                        convergence_checker=checker,
                                        x0_generator=generator,
                                        killing_conditions=hunters,
                                        share_best_solutions=True,  # Send good evals from one opt to another
                                        hunt_frequency=500,
                                        status_frequency=60,
                                        checkpoint_control=checkpointing,
                                        summary_files=3,
                                        is_log_detailed=False,
                                        visualisation=visualisation,
                                        visualisation_args=visualisation_args,
                                        force_terminations_after=-1,
                                        split_printstreams=True)

    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(lineno)d : %(name)s :: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger('glompo.manager')
    logger.addHandler(handler)
    logger.setLevel('INFO')

    result = manager.start_manager()

    print(f"Global min for Schwefel Function: {task.min_fx:.3E}")
    print("GloMPO minimum found:")
    print(result)
