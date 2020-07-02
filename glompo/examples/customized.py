from glompo import GloMPOManager
from glompo.benchmark_fncs import Michalewicz
from glompo.opt_selectors import ChainSelector
from glompo.convergence import MaxFuncCalls, TargetCost

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
        option will be individual set and explained.
    """

    # The Michaelewicz global optimization test function is a good example of where GloMPO can
    # outperform normal optimization.
    task = Michalewicz(dims=5)

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
    #      use 8 workers and population of 8. These are arguments required at CMA initialisation.
    cma_init_args = {'workers': 6, 'popsize': 6}
    # For Nevergrad we need only select the DE algorithm:
    ng_init_args = {'optimizer': 'DE'}

    # We can now setup the selector. Since we do not need to send any special arguments to optimizer.minimize we will
    # just setup with None. As mentioned above we will use CMA in the first half of the optimization and DE for the
    # other.
    selector = ChainSelector([(CMAOptimizer, cma_init_args, None),
                              (Nevergrad, ng_init_args, None)],
                             [max_calls/2])

    # Note the load balancing here. GloMPO will allow a fixed number of threads to be run. By default this is one less
    # than the number of CPUs available. If your machine has 32 cores, for example, than the manager will use 1 and
    # allow 31 to be used by the local optimizers. The 'workers' keyword we used for the optimizer earlier tells GloMPO
    # that each instance of CMA will use 6 of these slots. Thus GloMPO will start a maximum of 5 parallel CMA optimizers
    # in this run and one core will remain unused. This behaviour will change based on your machine's resources!
    #
    # If you want to fix the number of threads used regardless of the system resources, pass the optional max_jobs
    # argument during the manager initialisation.

    # BaseHunter objects are setup in a similar way to BaseChecker objects and control the conditions in which

    # The manager is setup using all GloMPO defaults in this case. Only the task, its box bounds and local
    # optimizers need be provided.
    manager = GloMPOManager(task=task,
                            optimizer_selector=selector,
                            bounds=task.bounds,
                            # OPTIONAL:
                            max_jobs=32)

    # To execute the minimization we simply run start_manager(). Note: by default GloMPO will not save any files but
    # this is available

    result = manager.start_manager()

    # Finally we print the selected minimum
    print(f"Global min for Michalewicz Function: {task.min_fx:.3E}")
    print("GloMPO minimum found:")
    print(result)
