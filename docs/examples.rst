********
Examples
********

GloMPO comes bundled with several examples to get you started. They can all be found in the ``examples``
directory of the package.

Minimal
*******

The :download:`minimal <../examples/minimal.py>` example configures the minimum number of GloMPO options (i.e. uses all
of its defaults) and demonstrates that very simple configurations are possible.

.. literalinclude:: ../examples/minimal.py
   :linenos:

The :class:`Michalewicz <glompo.benchmark_fncs.Michalewicz>` global optimization test function is a good
example of where GloMPO can outperform normal optimization.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 7

For this task we will use :class:`CMA-ES <glompo.optimizers.cmawrapper.CMAOptimizer>` which has good optimization
properties for many function classes. Optimizers are sent to GloMPO via
:class:`~glompo.opt_selectors.baseselector.BaseSelector` objects. These are code stubs which propose an optimizer type
and configuration to start when asked by the manager.

A very basic selector is :class:`CycleSelector <glompo.opt_selectors.cycle.CycleSelector>` which returns a rotating list
of optimizers when asked but can be used for just a single optimizer type.

Setting up any selector requires that a sequence of available optimizers be given to it during initialisation.
The elements in this list can take two forms:

#. Uninitiated :ref:`optimizer <Optimizers>` class.

#. Tuple of:

   #. Uninitiated :ref:`optimizer <Optimizers>` class;

   #. Dictionary of optional initialisation arguments;

   #. Dictionary of optional arguments passed to :meth:`~glompo.optimizers.baseoptimizer.BaseOptimizer.minimize`.

In this case we need to setup:

The initial :math:`\sigma` value:
   We choose this to be half the range of the bounds in each direction (in this case
   the bounds are equal in all directions). This value must be sent to
   :meth:`~glompo.optimizers.baseoptimizer.BaseOptimizer.minimize`.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 9-10

The number of parallel workers:
   CMA is a population based solver and uses multiple function evaluations per iteration; this is the population size.
   It can also use internal parallelization to evaluate each population member simultaneously; this is the number of
   workers or threads it can start. It is important that the user takes care of the load balancing at this point to
   ensure the most efficient performance. In this case we will use 1 worker and population of 6 (the function evaluation
   in this toy example is too fast to justify the overhead of multithreading or multiprocessing). These are arguments
   required at CMA initialisation.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 11

We can now setup the selector.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 12

Note the load balancing here. GloMPO will allow a fixed number of threads to be run. By default this is one less than
the number of CPUs available. If your machine has 32 cores for example than the manager will use 1 and allow 31 to be
used by the local optimizers. The :attr:`workers` keyword we used for the optimizer earlier tells GloMPO that each
instance of CMA will use 1 of these slots. Thus, GloMPO will start a maximum of 31 parallel CMA optimizers in this run.
Alternatively, if we had parallelized the function evaluations (by setting :attr:`workers` equal to 6) then 5 optimizers
would be started taking 6 slots each. In such a configuration one core of the 32 core machine would remain unused:
:math:`5\times6=30\text{optimizers} + 1\text{manager} = 31`.

If you want to fix the number of threads used regardless of the system resources, pass the optional
:attr:`~glompo.core.manager.GloMPOManager.max_jobs` argument during the manager initialisation.

The manager is setup using all GloMPO defaults in this case. Only the task, its box bounds and local optimizers need be
provided.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 14

To execute the minimization we simply run :meth:`~glompo.core.manager.GloMPOManager.start_manager`. Note: by default
GloMPO will not save any files but this is available.

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

The :download:`customized <../examples/customized.py>` example guides users through each of the options available to
configure the manager and will give the user a good overview of what is possible.

Nudging
*******

The :download:`nudging <../examples/customized.py>` example is broadly equivalent to the customized one, but
includes configuration settings for GloMPO to share information between optimizers in real-time. One should observe
a dramatic improvement in GloMPO's performance.

After working through the examples, the user is encouraged to read further in the documentation to get a proper
understanding of all of GloMPO's components.
