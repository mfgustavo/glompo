from glompo.benchmark_fncs import Michalewicz
from glompo.core.manager import GloMPOManager
from glompo.opt_selectors import CycleSelector
from glompo.optimizers.cmawrapper import CMAOptimizer

if __name__ == '__main__':
    """ In this example GloMPO will be run on a well known global optimization test function using all defaults.
     
        The :class:`Michaelewicz` global optimization test function is a good example of 
        where GloMPO can outperform normal optimization. 
    """
    task = Michalewicz(dims=5)

    """ For this task we will use :class:`CMA-ES <CMAOptimizer>` which has good optimization properties for many 
        function classes. Optimizers are sent to GloMPO via :class:`BaseSelector <Selectors>` objects. These are code 
        stubs which propose an optimizer type and configuration to start when asked by the manager.
    
        A very basic selector is :class:`CycleSelector` which returns a rotating list of optimizers when asked but can
        be used for just a single optimizer type.
    
        Setting up any selector requires that a list of available optimizers be given to it during initialisation.
        The elements in this list can take two forms:
        
        #. Uninitiated `optimizer <Optimizers>`_ class.
        
        #. Tuple of:
        
           #. Uninitiated `optimizer <Optimizers>`_ class;
           
           #. Dictionary of optional initialisation arguments;
           
           #. Dictionary of optional arguments passed to :meth:~`BaseOptimizer.minimize`
    
        In this case we need to setup:
        
        #. The initial sigma value. We choose this to be half the range of the bounds in each direction (in this case
           the bounds are equal in all directions). This value must be sent to optimizer.minimize
    """
    sigma = (task.bounds[0][1] - task.bounds[0][0]) / 2
    call_args = {'sigma0': sigma}
    """ 2. The number of parallel workers. CMA is a population based solver and uses multiple function evaluations per
           iteration; this is the population size. It can also use internal parallelization to evaluate each population
           member simultaneously; this is the number of workers or threads it can start. It is important that the user
           takes care of the load balancing at this point to ensure the most efficient performance. In this case we will
           use 1 worker and population of 6 (the function evaluation in this toy example is too fast to justify the
           overhead of multithreading or multiprocessing). These are arguments required at CMA initialisation.
    """
    init_args = {'workers': 1, 'popsize': 6}

    """ We can now setup the selector. """
    selector = CycleSelector((CMAOptimizer, init_args, call_args))

    """ Note the load balancing here. GloMPO will allow a fixed number of threads to be run. By default this is one less
        than the number of CPUs available. If your machine has 32 cores for example than the manager will use 1 and 
        allow 31 to be used by the local optimizers. The 'workers' keyword we used for the optimizer earlier tells 
        GloMPO that each instance of CMA will use 1 of these slots. Thus, GloMPO will start a maximum of 31 parallel 
        CMA optimizers in this run. Alternatively, if we had parallelized the function evaluations (by setting 'workers'
        equal to 6) then 5 optimizers would be started taking 6 slots each. In such a configuration one core of the
        32 core machine would remain unused: 5x6=30optimizers + 1manager = 31.
        
        If you want to fix the number of threads used regardless of the system resources, pass the optional max_jobs
        argument during the manager initialisation.
        
        The manager is setup using all GloMPO defaults in this case. Only the task, its box bounds and local
        optimizers need be provided.
    """
    manager = GloMPOManager.new_manager(task=task, bounds=task.bounds, opt_selector=selector)

    """ To execute the minimization we simply run start_manager(). Note: by default GloMPO will not save any files but
        this is available.
    """

    result = manager.start_manager()

    """ Finally we print the selected minimum """
    print(f"Global min for Michalewicz Function: {task.min_fx:.3E}")
    print("GloMPO minimum found:")
    print(result)
