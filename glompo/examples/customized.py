import logging
import sys

from glompo import GloMPOManager
from glompo.benchmark_fncs import Michalewicz
from glompo.convergence import MaxFuncCalls, TargetCost
from glompo.generators import ExploitExploreGenerator
from glompo.hunters import ParameterDistance, PseudoConverged, TypeHunter, ValueAnnealing
from glompo.opt_selectors import ChainSelector

try:
    from glompo.optimizers.cmawrapper import CMAOptimizer
except ModuleNotFoundError:
    raise ModuleNotFoundError("To run this example the cma package is required.")

try:
    from glompo.optimizers.nevergrad import Nevergrad
except ModuleNotFoundError:
    raise ModuleNotFoundError("To run this example the nevergrad package is required.")

if __name__ == '__main__':
    """ In this example GloMPO will be run on a well known global optimization test function but each configuration
        option will be individually set and explained.
    """

    # The Michaelewicz global optimization test function is a good example of where GloMPO can
    # outperform normal optimization. We add a delay to each evaluation to simulate the behaviour of a real
    # optimization test function.
    task = Michalewicz(dims=5, delay=0.02)

    # Convergence of the GloMPO manager is controlled by BaseChecker objects. This are small classes which define a
    # single termination condition. These classes can then be easily combined to create sophisticated termination
    # conditions using & and | symbolics.
    #
    # In this case we would like the optimizer to run for a fixed number of iterations or stop if the global minimum is
    # found. Of course we would not know the global minimum in typical problems but we do in this case.
    max_calls = 100000  # Imposed iteration budget
    checker = MaxFuncCalls(max_calls) | TargetCost(task.min_fx)  # Combined two conditions into a single stop criteria.

    # For this task we will use two classes of optimizers. CMA-ES is good for global exploration and identifying areas
    # of interest but performs poorly when serialised i.e. one iteration per evaluation. Differential Evolution also
    # uses some form of covariance adaptation and is very quick in high dimension. We will run CMA-ES for the first half
    # of our iteration budget and then replace dead CMA optimizers with several DE optimizers in the latter half of the
    # optimization. The intention is to use CMA to identify areas of interest and then multiple copies of DE to
    # thoroughly explore them.
    #
    # Optimizers are sent to GloMPO via BaseSelector objects. These are code stubs which propose an optimizer type and
    # configuration to start when asked by the manager.
    #
    # For this case we need the ChainSelector which bases its return on the number of function evaluations already
    # carried out.
    #
    # Setting up any selector requires that a list of available optimizers be given to it during initialisation.
    # The elements in this list can take two forms:
    #    1. Uninitiated optimizer class
    #       OR
    #    2. Tuple of:
    #       a. Uninitiated optimizer class
    #       b. Dictionary of optional initialisation arguments
    #       c. Dictionary of optional arguments passed to the optimizer.minimize argument
    #
    # For CMA we need to setup:
    #   1. The initial sigma value. We choose this to be half the range of the bounds in each direction (in this case
    #      the bounds are equal in all directions):
    sigma = (task.bounds[0][1] - task.bounds[0][0]) / 2
    #   2. The number of parallel workers. CMA is a population based solver and uses multiple function evaluations per
    #      iteration; this is the population size. It can also use internal parallelization to evaluate each population
    #      member simultaneously; this is the number of workers or threads it can start. It is important that the user
    #      takes care of the load balancing at this point to ensure the most efficient performance. In this case we will
    #      use 6 workers and population of 6. These are arguments required at CMA initialisation.
    cma_init_args = {'workers': 6, 'popsize': 6}
    # For Nevergrad we need only select the DE algorithm:
    ng_init_args = {'optimizer': 'DE'}

    # We can now setup the selector. Since we do not need to send any special arguments to optimizer.minimize we will
    # just setup the call arguments with None. As mentioned above we will use CMA in the first half of the optimization
    # and DE for the other.
    selector = ChainSelector(avail_opts=[(CMAOptimizer, cma_init_args, None),
                                         (Nevergrad, ng_init_args, None)],
                             fcall_thresholds=[max_calls/2])

    # Note the load balancing here. GloMPO will allow a fixed number of threads to be run. By default this is one less
    # than the number of CPUs available. If your machine has 32 cores, for example, than the manager will use 1 and
    # allow 31 to be used by the local optimizers. The 'workers' keyword we used for the optimizer earlier tells GloMPO
    # that each instance of CMA will use 6 of these slots. Thus GloMPO will start a maximum of 5 parallel CMA optimizers
    # in this run and one core will remain unused. This behaviour will change based on your machine's resources!
    #
    # If you want to fix the number of threads used regardless of the system resources, pass the optional max_jobs
    # argument during the manager initialisation.
    max_jobs = 32  # OR os.cpu_count()

    # BaseHunter objects are setup in a similar way to BaseChecker objects and control the conditions in which
    # optimizers are shutdown by the manager. Each hunter is individually documented in ..hunters.
    #
    # In this example we want two different sets of conditions:
    #   1. We want to kill CMA optimizers early and not let them spend many iterations converging. We also want to
    #      kill optimizers that come too close together so that we are not wasting iterations exploring the same region
    #      repeatedly:
    cma_killers = TypeHunter(CMAOptimizer) & (PseudoConverged(calls=1000, tol=0.05) |
                                              ParameterDistance(bounds=task.bounds, relative_distance=0.05,
                                                                test_all=True))
    #   2. We want to give DE optimizers more time to find the answer, we also want to avoid killing optimizers with
    #      function values that are very close together even though they are far apart.
    de_killers = TypeHunter(Nevergrad) & (PseudoConverged(calls=2000, tol=0.01) & ValueAnnealing() |
                                          ParameterDistance(bounds=task.bounds, relative_distance=0.01, test_all=True))

    all_killers = cma_killers | de_killers
    # Note: Hunters and checkers are evaluated lazily this means that in x | y, y will not be evaluated if x is True and
    # in x & y, y will not be evaluated if x is False.

    # BaseSelector objects select which optimizers to start but BaseGenerator objects select a point in parameter space
    # where to start them.
    #
    # In this example we will use the ExploreExploitGenerator which starts optimizers at random locations early in the
    # optimization but then starts them near known minima towards the end of the optimization budget.
    generator = ExploitExploreGenerator(bounds=task.bounds, max_func_calls=max_calls, focus=0.3)

    # GloMPO supports running the optimizers both as threads and processes. Processes are preferred and the default
    # since they circumvent the python Global Interpreter Lock but threads can also be used for tasks that are not
    # multiprocessing safe. In this example we will use processes.
    backend = 'processes'

    # GloMPO includes a dynamic scope allowing one to watch the optimization progress in real-time using a graphic.
    # This can be very helpful when configuring GloMPO and the results can be saved as movie files. This functionality
    # requires an interactive version of matplotlib and ffmpeg installed on the system.
    # This is turned on by default for this example but if the script fails simply set visualisation to False to bypass
    # it.
    visualisation = True
    visualisation_args = {'record_movie': True,
                          'x_range': (0, max_calls),
                          'y_range': None,
                          'log_scale': False,
                          'events_per_flush': 300,
                          'interactive_mode': True,
                          'writer_kwargs': {'fps': 4},
                          'movie_kwargs': {'outfile': 'distance_demo.mp4',
                                           'dpi': 200}}

    # For buggy tasks which are liable to fail or produce extreme results, it is possible that optimizers can get stuck
    # and simply never return. If this is a risk that we can send a timeout condition after which the manager will force
    # them to crash. Note that this will not work on threaded backends. In this example this is not needed so we leave
    # the default as -1.
    force_terminations = -1

    # All arguments are now fed to the manager initialisation
    manager = GloMPOManager(task=task,
                            optimizer_selector=selector,
                            bounds=task.bounds,
                            # OPTIONAL:
                            working_dir="customized_example_outputs",  # Directory to save result files
                            overwrite_existing=True,  # Deletes any previous results in working_dir
                            max_jobs=max_jobs,  # Maximum number of threads optimizers can use
                            task_args=None,  # Optional arguments needed for task.__call__()
                            task_kwargs=None,    # Optional keyword arguments needed for task.__call__()
                            backend=backend,
                            convergence_checker=checker,
                            x0_generator=generator,
                            killing_conditions=all_killers,
                            hunt_frequency=500,  # Task eval. frequency manager applies kill conditions.
                            summary_files=3,  # Maximum file saving, all logs and printstreams are retained
                            visualisation=visualisation,
                            visualisation_args=visualisation_args,
                            force_terminations_after=-1,
                            split_printstreams=True)  # Automatically send print statements from opts to different files

    # GloMPO contains built-in logging statements throughout the library. These will not show up by default but can be
    # accessed if desired. In fact intercepting the INFO level statements from the manager creates a nice progress
    # stream from the optimization; we will set this up here. Consult the README for more information.
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(lineno)d : %(name)s :: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger('glompo.manager')
    logger.addHandler(handler)
    logger.setLevel('INFO')

    # To execute the minimization we simply run start_manager().
    result = manager.start_manager()

    # Finally we print the selected minimum
    print(f"Global min for Michalewicz Function: {task.min_fx:.3E}")
    print("GloMPO minimum found:")
    print(result)
