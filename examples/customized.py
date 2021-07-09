import logging
import sys

from glompo.benchmark_fncs import Schwefel
from glompo.convergence import MaxFuncCalls, TargetCost
from glompo.core.checkpointing import CheckpointingControl
from glompo.core.manager import GloMPOManager
from glompo.generators import RandomGenerator
from glompo.hunters import BestUnmoving, EvaluationsUnmoving, ParameterDistance, ValueAnnealing
from glompo.opt_selectors import CycleSelector

try:
    from glompo.optimizers.cmawrapper import CMAOptimizer
except ModuleNotFoundError:
    raise ModuleNotFoundError("To run this example the cma package is required.")

if __name__ == '__main__':
    """ GloMPO contains built-in logging statements throughout the library. These will not show up by default but can be
        accessed if desired. In fact intercepting the INFO level statements from the manager creates a nice progress
        stream from the optimization; we will set this up here. Consult the README for more information.
    """
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(lineno)d : %(name)s :: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger('glompo.manager')
    logger.addHandler(handler)
    logger.setLevel('INFO')

    """ In this example GloMPO will be run on a well known global optimization test function but each configuration
        option will be individually set and explained.
    
        The Schwefel global optimization test function is a good example of where GloMPO can
        outperform normal optimization.
    """
    task = Schwefel(dims=20)

    """ Convergence of the GloMPO manager is controlled by BaseChecker objects. This are small classes which define a
        single termination condition. These classes can then be easily combined to create sophisticated termination
        conditions using & and | symbolics.
        
        In this case we would like the optimizer to run for a fixed number of iterations or stop if the global minimum
        is found. Of course we would not know the global minimum in typical problems but we do in this case.
    """
    max_calls = 100000  # Imposed iteration budget
    checker = MaxFuncCalls(max_calls) | TargetCost(task.min_fx)  # Combined two conditions into a single stop criteria.

    """ For this task we will use a CMA-ES optimizer.
        
        Optimizers are sent to GloMPO via BaseSelector objects. These are code stubs which propose an optimizer type and
        configuration to start when asked by the manager.
        
        A very basic selector is CycleSelector which returns a rotating list of optimizers when asked but can be used 
        for just a single optimizer type.
        
        Setting up any selector requires that a list of available optimizers be given to it during initialisation.
        The elements in this list can take two forms:
           1. Uninitiated optimizer class
           2. Tuple of:
              a. Uninitiated optimizer class
              b. Dictionary of optional initialisation arguments
              c. Dictionary of optional arguments passed to the optimizer.minimize argument
        
        In this case we need to setup:
          1. The initial sigma value. We choose this to be half the range of the bounds in each direction (in this case
             the bounds are equal in all directions). This value must be sent to optimizer.minimize
    """
    sigma = (task.bounds[0][1] - task.bounds[0][0]) / 2
    call_args = {'sigma0': sigma}
    """ 2. The number of parallel workers. CMA is a population based solver and uses multiple function evaluations per
           iteration; this is the population size. It can also use internal parallelization to evaluate each population
           member simultaneously; this is the number of workers or threads it can start. It is important that the user
           takes care of the load balancing at this point to ensure the most efficient performance. In this case we will
           use 1 worker and population of 12.
           
           These are arguments required at CMA initialisation.
    """
    init_args = {'workers': 1, 'popsize': 12}

    """ We can now setup the selector. """
    selector = CycleSelector((CMAOptimizer, init_args, call_args))

    """ Note the load balancing here. GloMPO will allow a fixed number of threads to be run. By default this is one less
        than the number of CPUs available. If your machine has 32 cores, for example, than the manager will use 1 and
        allow 31 to be used by the local optimizers. The 'workers' keyword we used for the optimizer earlier determines
        how many core GloMPO will assign to each worker.
        
        If you want to fix the number of threads used regardless of the system resources, pass the optional max_jobs
        argument during the manager initialisation.
    """
    max_jobs = 10  # OR os.cpu_count()

    """ BaseHunter objects are setup in a similar way to BaseChecker objects and control the conditions in which
        optimizers are shutdown by the manager. Each hunter is individually documented in ..hunters.
        
        In this example we will use a hunting set which has proven effective on several problems:
    """
    hunters = (EvaluationsUnmoving(100, 0.01) &  # Kill optimizers which are incorrectly focussing
               ValueAnnealing(0.10) |  # Keep competitive optimizers alive
               BestUnmoving(int(max_calls / 15), 0.2) |  # Kill optimizers that go nowhere for a long time
               ParameterDistance(task.bounds, 0.05))  # Kill optimizers that go to the same minimum
    """ Note: Hunters and checkers are evaluated lazily this means that in x | y, y will not be evaluated if x is True
        and in x & y, y will not be evaluated if x is False.

        BaseSelector objects select which optimizers to start but BaseGenerator objects select a point in parameter 
        space where to start them.
        
        In this example we will use the RandomGenerator which starts optimizers at random locations.
    """
    generator = RandomGenerator(task.bounds)

    """ GloMPO supports running the optimizers both as threads and processes. Processes are preferred and the default
        since they circumvent the Python Global Interpreter Lock but threads can also be used for tasks that are not
        multiprocessing safe. In this example we will use processes.
        
        It is highly recommended that the user familiarize themselves with Python's behavior in this regard! If all 
        computations are performed within Python than multithreading will NOT result in the distribution of calculations
        over more than one core. The interested user should search for the 'Python Global Interpreter Lock' for more
        information. 
    """
    backend = 'processes'

    """ GloMPO includes a dynamic scope allowing one to watch the optimization progress in real-time using a graphic.
        This can be very helpful when configuring GloMPO and the results can be saved as movie files. This functionality
        requires an interactive version of matplotlib and ffmpeg installed on the system.
        
        This is turned on by default for this example but if the script fails simply set visualisation to False to 
        bypass it. Note also that the scope is very helpful for preliminary configuration but is slow due to matplotlib
        limitations and should not be used during production runs.
    """
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

    """ For buggy tasks which are liable to fail or produce extreme results, it is possible that optimizers can get 
        stuck and simply never return. If this is a risk that we can send a timeout condition after which the manager 
        will force them to crash. Note that this will not work on threaded backends. In this example this is not needed
        so we leave the default as -1.
    """
    force_terminations = -1

    """ GloMPO supports checkpointing. This means that its state can be persisted to file during an optimization and 
        this checkpoint file can be loaded by another GloMPO instance to resume the optimization from that point. 
        Checkpointing options are configured through a CheckpointingControl instance. In this case we will produce a 
        checkpoint called 'customized_completed_<DATE>_<TIME>.tar.gz' once the task has converged.
    """
    checkpointing = CheckpointingControl(checkpoint_at_conv=True,
                                         naming_format='customized_completed_%(date)_%(time).tar.gz',
                                         checkpointing_dir="customized_example_outputs")

    """ All arguments are now fed to the manager initialisation """
    manager = GloMPOManager.new_manager(task=task,
                                        bounds=task.bounds,
                                        opt_selector=selector,
                                        working_dir="customized_example_outputs",  # Dir in which files are saved
                                        overwrite_existing=True,  # Replaces existing files in working directory
                                        max_jobs=max_jobs,
                                        backend=backend,
                                        convergence_checker=checker,
                                        x0_generator=generator,
                                        killing_conditions=hunters,
                                        share_best_solutions=False,  # Send good evals from one opt to another
                                        hunt_frequency=500,  # Function evaluations between hunts
                                        status_frequency=60,  # Interval in seconds in which a status message is logged
                                        checkpoint_control=checkpointing,
                                        summary_files=3,  # Controls the level of output produced
                                        is_log_detailed=False,  # Functions can produce extra info which can be logged
                                        visualisation=visualisation,
                                        visualisation_args=visualisation_args,
                                        force_terminations_after=-1,
                                        split_printstreams=True)  # Autosend print statements from opts to files

    """ To execute the minimization we simply run start_manager(). """
    result = manager.start_manager()

    """ Finally we print the selected minimum. """
    print(f"Global min for Schwefel Function: {task.min_fx:.3E}")
    print("GloMPO minimum found:")
    print(result)
