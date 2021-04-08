from glompo import GloMPOManager
from glompo.benchmark_fncs import Michalewicz
from glompo.opt_selectors import CycleSelector
from glompo.optimizers.cmawrapper import CMAOptimizer

if __name__ == '__main__':
    """ In this example GloMPO will be run on a well known global optimization test function using all defaults. """

    # The Michaelewicz global optimization test function is a good example of where GloMPO can
    # outperform normal optimization.
    task = Michalewicz(dims=5)

    # For this task we will use CMA-ES which has good optimization properties for many function classes.
    # Optimizers are sent to GloMPO via BaseSelector objects. These are code stubs which propose an optimizer type and
    # configuration to start when asked by the manager.
    #
    # A very basic selector is CycleSelector which returns a rotating list of optimizers when asked but can be used for
    # just a single optimizer type.
    #
    # Setting up any selector requires that a list of available optimizers be given to it during initialisation.
    # The elements in this list can take two forms:
    #    1. Uninitiated optimizer class
    #    2. Tuple of:
    #       a. Uninitiated optimizer class
    #       b. Dictionary of optional initialisation arguments
    #       c. Dictionary of optional arguments passed to the optimizer.minimize argument
    #
    # In this case we need to setup:
    #   1. The initial sigma value. We choose this to be half the range of the bounds in each direction (in this case
    #      the bounds are equal in all directions). This value must be sent to optimizer.minimize
    sigma = (task.bounds[0][1] - task.bounds[0][0]) / 2
    call_args = {'sigma0': sigma}
    #   2. The number of parallel workers. CMA is a population based solver and uses multiple function evaluations per
    #      iteration; this is the population size. It can also use internal parallelization to evaluate each population
    #      member simultaneously; this is the number of workers or threads it can start. It is important that the user
    #      takes care of the load balancing at this point to ensure the most efficient performance. In this case we will
    #      use 6 workers and population of 6. These are arguments required at CMA initialisation.
    init_args = {'workers': 6, 'popsize': 6}

    # We can now setup the selector.
    selector = CycleSelector([(CMAOptimizer, init_args, call_args)])

    # Note the load balancing here. GloMPO will allow a fixed number of threads to be run. By default this is one less
    # than the number of CPUs available. If your machine has 32 cores for example than the manager will use 1 and allow
    # 31 to be used by the local optimizers. The 'workers' keyword we used for the optimizer earlier tells GloMPO that
    # each instance of CMA will use 6 of these slots. Thus GloMPO will start a maximum of 5 parallel CMA optimizers in
    # this run and one core will remain unused. This behaviour will change based on your machine's resources!
    #
    # If you want to fix the number of threads used regardless of the system resources, pass the optional max_jobs
    # argument during the manager initialisation.

    # The manager is setup using all GloMPO defaults in this case. Only the task, its box bounds and local
    # optimizers need be provided.
    manager = GloMPOManager.new_manager(task=task, bounds=task.bounds, opt_selector=selector, max_jobs=32)

    # To execute the minimization we simply run start_manager(). Note: by default GloMPO will not save any files but
    # this is available

    result = manager.start_manager()

    # Finally we print the selected minimum
    print(f"Global min for Michalewicz Function: {task.min_fx:.3E}")
    print("GloMPO minimum found:")
    print(result)
