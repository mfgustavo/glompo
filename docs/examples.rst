********
Examples
********

GloMPO comes bundled with several examples to get you started. They can all be found in the ``examples`` directory of the package. The examples are best done in the order presented here. After working them, the user is encouraged to read further in the documentation to get a proper understanding of all of GloMPO's components.

.. contents:: Contents
   :local:

Minimal
*******

The :download:`minimal <../examples/minimal.py>` example configures the minimum number of GloMPO options (i.e. uses all of its defaults) and demonstrates that very simple configurations are possible.

.. literalinclude:: ../examples/minimal.py
   :linenos:

The :class:`Michalewicz <.Michalewicz>` global optimization test function is a good example of where GloMPO can outperform normal optimization.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 7

For this task we will use :class:`CMA-ES <.CMAOptimizer>` which has good optimization properties for many function classes. Optimizers are sent to GloMPO via :class:`.BaseSelector` objects. These are code stubs which propose an optimizer type and configuration to start when asked by the manager.

A very basic selector is :class:`CycleSelector <.CycleSelector>` which returns a rotating list of optimizers when asked but can be used for just a single optimizer type.

Setting up any selector requires that a sequence of available optimizers be given to it during initialisation. The elements in this list can take two forms:

#. Uninitiated :ref:`optimizer <Optimizers>` class.

#. Tuple of:

   #. Uninitiated :ref:`optimizer <Optimizers>` class;

   #. Dictionary of optional initialisation arguments;

   #. Dictionary of optional arguments passed to :meth:`.BaseOptimizer.minimize`.

In this case we need to setup:

The initial :math:`\sigma` value:
   We choose this to be half the range of the bounds in each direction (in this case the bounds are equal in all directions). This value must be sent to :meth:`~.BaseOptimizer.minimize`.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 9-10

The number of parallel workers:
   CMA is a population based solver and uses multiple function evaluations per iteration; this is the population size. It can also use internal parallelization to evaluate each population member simultaneously; this is the number of workers or threads it can start. It is important that the user takes care of the load balancing at this point to ensure the most efficient performance. In this case we will use 1 worker and population of 6 (the function evaluation in this toy example is too fast to justify the overhead of multithreading or multiprocessing). These are arguments required at CMA initialisation.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 11

We can now setup the selector.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 12

Note the load balancing here. GloMPO will allow a fixed number of threads to be run. By default this is one less than the number of CPUs available. If your machine has 32 cores for example than the manager will use 1 and allow 31 to be used by the local optimizers. :code:`'workers'` keyword we used for the optimizer earlier tells GloMPO that each instance of CMA will use 1 of these slots. Thus, GloMPO will start a maximum of 31 parallel CMA optimizers in this run. Alternatively, if we had parallelized the function evaluations (by setting :code:`'workers'` equal to 6) then 5 optimizers would be started taking 6 slots each. In such a configuration one core of the 32 core machine would remain unused: :math:`5\times6=30\text{optimizers} + 1\text{manager} = 31`.

If you want to fix the number of threads used regardless of the system resources, pass the optional :attr:`~.GloMPOManager.max_jobs` argument during the manager initialisation.

The manager is setup using all GloMPO defaults in this case. Only the task, its box bounds and local optimizers need be provided.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 14

To execute the minimization we simply run :meth:`.GloMPOManager.start_manager`. Note: by default GloMPO will not save any files but this is available.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 15

Finally we print the selected minimum

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 17-19

Customized
**********

The :download:`customized <../examples/customized.py>` example guides users through each of the options available to configure the manager and will give the user a good overview of what is possible.

.. literalinclude:: ../examples/customized.py
   :linenos:

GloMPO contains built-in logging statements throughout the library. These will not show up by default but can be accessed if desired. In fact intercepting the `logging.INFO <https://docs.python.org/3.6/library/logging.html?#logging-levels>`_ level statements from the manager creates a nice progress stream from the optimization; we will set this up here. See :ref:`Logging Messages` for more information.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 18-25

In this example GloMPO will be run on a well known global optimization test function but each configuration option will be individually set and explained.

The :class:`~.benchmark_fncs.Schwefel` global optimization test function is a good example of where GloMPO can outperform normal optimization.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 27

Convergence of the GloMPO manager is controlled by :class:`.BaseChecker` objects. These are small classes which define a single termination condition. These classes can then be easily combined to create sophisticated termination conditions using :code:`&` and :code:`|` symbolics.

In this case we would like the optimizer to run for a fixed number of iterations or stop if the global minimum is found. Of course we would not know the global minimum in typical problems but we do in this case.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 29-30

We will configure the optimizers as was done in the :ref:`Minimal` example:

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 32-35

The :ref:`Minimal` example discussed the importance of load balancing. In this example we will override the default number of slots and limit the manager to 10:

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 37

:class:`.BaseHunter` objects are setup in a similar way to :class:`.BaseChecker` objects and control the conditions in which optimizers are shutdown by the manager. Each hunter is individually documented :ref:`here <Included Hunters>`.

In this example we will use a hunting set which has proven effective on several problems:

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 39-42

.. note::

   :class:`.BaseHunter` and :class:`.BaseChecker` are evaluated lazily this means that in :code:`x | y`, :code:`y` will
   not be evaluated if :code:`x` is :obj:`True` and in :code:`x & y`, :code:`y` will not be evaluated if :code:`x` is
   :obj:`False`.

:class:`.BaseSelector` objects select which optimizers to start but :class:`.BaseGenerator` objects select a point in parameter space where to start them.

In this example we will use the :class:`.RandomGenerator` which starts optimizers at random locations.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 44

GloMPO supports running the optimizers both as threads and processes. Processes are preferred and the default since they circumvent the `Python Global Interpreter Lock <https://docs.python.org/3.6/glossary.html#term-global-interpreter-lock>`_ but threads can also be used for tasks that are not multiprocessing safe. In this example we will use processes.

.. attention::

   It is highly recommended that the user familiarize themselves with Python's behavior in this regard! If all
   computations are performed within Python than multithreading will NOT result in the distribution of calculations
   over more than one core.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 46

GloMPO includes a dynamic scope allowing one to watch the optimization progress in real-time using a graphic. This can be very helpful when configuring GloMPO and the results can be saved as movie files. This functionality requires :doc:`matplotlib <matplotlib:index>` and `ffmpeg <ffmpeg.org>`_ installed on the system.

This is turned on for this example but if the script fails simply set :code:`visualisation` to :obj:`False` to bypass it. Note also that the scope is very helpful for preliminary configuration but is slow due to :doc:`matplotlib <matplotlib:index>`\\'s limitations and should not be used during production runs.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 48-57

For buggy tasks which are liable to fail or produce extreme results, it is possible that optimizers can get stuck and simply never return. If this is a risk that we can send a timeout condition after which the manager will force them to crash. Note that this will not work on threaded backends. In this example this is not needed so we leave the default as -1.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 59

GloMPO supports checkpointing. This means that its state can be persisted to file during an optimization and this checkpoint file can be loaded by another GloMPO instance to resume the optimization from that point. Checkpointing options are configured through a :class:`.CheckpointingControl` instance. In this case we will produce a checkpoint called ``customized_completed_<DATE>_<TIME>.tar.gz`` once the task has converged.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 61-63

All arguments are now fed to the manager initialisation. Other settings which did not warrant detailed discussion above are commented upon below:

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 65-84

To execute the minimization we simply run :meth:`.GloMPOManager.start_manager`.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 86

Finally we print the selected minimum.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 88-90

Nudging
*******

The :download:`nudging <../examples/nudging.py>` example is a variation of the :ref:`Customized` one. GloMPO will be run on the same task with virtually the same configuration, but in this case good iterations will be shared between optimizers. The optimizers, in turn, will use this information to accelerate their convergence. The user should see a marked improvement in GloMPO's performance. Only two modifications to the :ref:`Customized` example are necessary:

In this case we tell CMA-ES to accept suggestions from the manager and sample these points once every 10 iterations.

.. literalinclude:: ../examples/nudging.py
   :linenos:
   :lineno-match:
   :lines: 34

The hunting must be reconfigured slightly to better suit the new optimization behavior:

.. literalinclude:: ../examples/nudging.py
   :linenos:
   :lineno-match:
   :lines: 39

Managing Two-Level Algorithms
*****************************

The :download:`twolevel <../examples/twolevel.py>` script provides a simple demonstration of GloMPO managing a popular two-level algorithm; basin-hopping. By 'two-level' algorithm we mean metaheuristics which include periodic local searches such as SciPy's `basin-hopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy-optimize-basinhopping>`_ or `dual annealing <https://docs.scipy.org/doc/scipy//reference/reference/generated/scipy.optimize.dual_annealing.html#scipy-optimize-dual-annealing>`_ algorithms.

These algorithms typically have an upper level routine (usually a Monte-Carlo jump) which selects points to evaluate. Local search routines are then started at these points. One can configure GloMPO to manage the overall strategy by launching instances of routines as its children (see :class:`.ScipyOptimizeWrapper`).

In this example, however, we demonstrate a different approach. Here the 'upper' level algorithm (which chooses where optimizers are started) is used as the :ref:`Generator <Generators>`, while the local searches are started as its children.

This is a proof of concept, showing how GloMPO's management and supervision aspects can be brought into existing optimization strategies without requiring a large amount of reimplementation.

.. literalinclude:: ../examples/twolevel.py
   :linenos:

In this example, we will use the :class:`.LennardJones` problem. The minimization problem seeks to optimally arrange :math:`N` atoms in :math:`d`-dimensional space which minimizes the overall energy (balances attractive and repulsive forces). We use a shifted and version so that all values are postive and can be conveniently displayed on the output plots.

.. literalinclude:: ../examples/twolevel.py
   :linenos:
   :lineno-match:
   :lines: 18-20

.. literalinclude:: ../examples/twolevel.py
   :linenos:
   :lineno-match:
   :lines: 33

The optimization will be terminated when the minimum is found or the function has been called 4000 times:

.. literalinclude:: ../examples/twolevel.py
   :linenos:
   :lineno-match:
   :lines: 35-36

We will use the :class:`.BasinHoppingGenerator` to pick points in the same way that the `basin-hopping <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy-optimize-basinhopping>`_ algorithm does.

.. literalinclude:: ../examples/twolevel.py
   :linenos:
   :lineno-match:
   :lines: 38

The managed optimizers will be instances of the BFGS algorithm used by the full routine:

.. literalinclude:: ../examples/twolevel.py
   :linenos:
   :lineno-match:
   :lines: 40-42

We can use a simple hunting scheme in this example which simply terminates optimizers which are stagnating at levels worse than those previously seen.

.. literalinclude:: ../examples/twolevel.py
   :linenos:
   :lineno-match:
   :lines: 46

The rest of the script follows the pattern of the previous examples. The components can now be given to the manager and run.

ReaxFF
******

.. attention::

   This example requires the `Amsterdam Modelling Suite <www.scm.com>`_ be present on your system with licenses for ReaxAMS and ParAMS.

   This example is also quite expensive and should be performed on an HPC node.

The :download:`reaxff <../examples/reaxff.py>` script provides an example of GloMPO configured to manage several parallel CMA-ES reparameterizations of the disulfide training set designed by `Muller et al. (2015) <https://doi.org/10.1002/jcc.23966>`_.

.. literalinclude:: ../examples/reaxff.py
   :linenos:

We begin by setting up the cost function from the `trainset.in`, `geo` and `ffield` files published with the paper. These are included in GloMPO's `tests/_test_inputs` directory:

.. literalinclude:: ../examples/reaxff.py
   :linenos:
   :lineno-match:
   :lines: 32

The optimization will be time-limited in this case to 24hrs:

.. literalinclude:: ../examples/reaxff.py
   :linenos:
   :lineno-match:
   :lines: 38

Resource balancing is critically important in this case since the optimization is expensive. Generally the cost function evaluation time is a function of the parameters being tested since geometry optimizations take longer to converge for some parameter combinations.

Thus, one can set:

   ``workers = popsize``
      This evaluates very quickly, but has the worst computational efficiency as many cores remain idle while waiting for the slowest evaluation.

   ``workers = 1``
      This evaluates the cost function sequentially and is the most computationally efficient (no idling time), but requires the most wall time since there is no parallelization.

It is often best to choose a balance between these considerations and factor in the availability of computational resources and the number of parallel optimizations one would like to run.

In this example, we assume a 24 core machine and would like to run 6 optimizers in parallel. This suggests that each optimizer should run 4 function evaluations in parallel; a suitable load balancing compromise. We select a population size of 12 (multiple of 4) to further optimize computational efficiency.

We use the CMA-ES algorithm which has been shown to perform well on ReaxFF reparameterizations, and select a moderate :math:`\sigma` value.

.. literalinclude:: ../examples/reaxff.py
   :linenos:
   :lineno-match:
   :lines: 40-45

.. attention::

   It is important that you rebalance these parameters according to your available resources before running the example.

We will keep hunting simple in this example, and shutdown optimizers which begin to converge towards points which are worse than those already seen. You could also consider including conditions based on the status of a validation set, or some other measure of overfitting. If you are testing a different training set, it may be helpful to include a condition to terminate optimizers which do not start to converge after a long time.

.. literalinclude:: ../examples/reaxff.py
   :linenos:
   :lineno-match:
   :lines: 47

In this example, we will look for parameter sets near the incumbent, as other good values are likely in the same region. It is possible to do a more exploratory search by using :class:`.RandomGenerator` and larger :math:`\sigma` value as was done in the :ref:`Customized` example. This creates the possibility of finding very different parameter sets, but may end up being more expensive as the optimizers explore non-physical and instable parameters.

.. literalinclude:: ../examples/reaxff.py
   :linenos:
   :lineno-match:
   :lines: 49-51

``'threads'`` must be used for the ``backend`` when working with ParAMS since it does not support multiprocessing at this level. However, actual evaluations of the cost functions will be spun off by processes at the PLAMS level and not be subject to the `Python GIL <https://docs.python.org/3.6/glossary.html#term-global-interpreter-lock>`_.

.. literalinclude:: ../examples/reaxff.py
   :linenos:
   :lineno-match:
   :lines: 53

A checkpoint is configured for the end of the optimization. If you would like to continue the optimization further, this file can be used to restart the optimization for the final state.

.. literalinclude:: ../examples/reaxff.py
   :linenos:
   :lineno-match:
   :lines: 55-57

The rest of the script follows the pattern of the previous examples. The components can now be given to the manager and run.
