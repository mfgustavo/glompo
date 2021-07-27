********
Examples
********

GloMPO comes bundled with several examples to get you started. They can all be found in the ``examples``
directory of the package. After working through the examples, the user is encouraged to read further in the
documentation to get a proper understanding of all of GloMPO's components.

.. _Minimal:

Minimal
*******

The :download:`minimal <../examples/minimal.py>` example configures the minimum number of GloMPO options (i.e. uses all
of its defaults) and demonstrates that very simple configurations are possible.

.. literalinclude:: ../examples/minimal.py
   :linenos:

The :class:`Michalewicz <.Michalewicz>` global optimization test function is a good example of where GloMPO can
outperform normal optimization.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 7

For this task we will use :class:`CMA-ES <.CMAOptimizer>` which has good optimization properties for many function
classes. Optimizers are sent to GloMPO via :class:`.BaseSelector` objects. These are code stubs which propose an
optimizer type and configuration to start when asked by the manager.

A very basic selector is :class:`CycleSelector <.CycleSelector>` which returns a rotating list of optimizers when asked
but can be used for just a single optimizer type.

Setting up any selector requires that a sequence of available optimizers be given to it during initialisation.
The elements in this list can take two forms:

#. Uninitiated :ref:`optimizer <Optimizers>` class.

#. Tuple of:

   #. Uninitiated :ref:`optimizer <Optimizers>` class;

   #. Dictionary of optional initialisation arguments;

   #. Dictionary of optional arguments passed to :meth:`.BaseOptimizer.minimize`.

In this case we need to setup:

The initial :math:`\sigma` value:
   We choose this to be half the range of the bounds in each direction (in this case
   the bounds are equal in all directions). This value must be sent to :meth:`~.BaseOptimizer.minimize`.

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
used by the local optimizers. :code:`'workers'` keyword we used for the optimizer earlier tells GloMPO
that each instance of CMA will use 1 of these slots. Thus, GloMPO will start a maximum of 31 parallel CMA optimizers in
this run. Alternatively, if we had parallelized the function evaluations (by setting :code:`'workers'` equal to 6) then
5 optimizers would be started taking 6 slots each. In such a configuration one core of the 32 core machine would remain
unused: :math:`5\times6=30\text{optimizers} + 1\text{manager} = 31`.

If you want to fix the number of threads used regardless of the system resources, pass the optional
:attr:`~.GloMPOManager.max_jobs` argument during the manager initialisation.

The manager is setup using all GloMPO defaults in this case. Only the task, its box bounds and local optimizers need be
provided.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 14

To execute the minimization we simply run :meth:`.GloMPOManager.start_manager`. Note: by default GloMPO will not save
any files but this is available.

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 15

Finally we print the selected minimum

.. literalinclude:: ../examples/minimal.py
   :linenos:
   :lineno-match:
   :lines: 17-19

.. _Customized:

Customized
**********

The :download:`customized <../examples/customized.py>` example guides users through each of the options available to
configure the manager and will give the user a good overview of what is possible.

.. literalinclude:: ../examples/customized.py
   :linenos:

GloMPO contains built-in logging statements throughout the library. These will not show up by default but can be
accessed if desired. In fact intercepting the
`logging.INFO <https://docs.python.org/3.6/library/logging.html?highlight=logging%20info#logging-levels>`_ level
statements from the manager creates a nice progress stream from the optimization; we will set this up here.
See :ref:`Logging Messages` for more information.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 18-25

In this example GloMPO will be run on a well known global optimization test function but each configuration option will
be individually set and explained.

The :class:`~.benchmark_fncs.Schwefel` global optimization test function is a good example of where GloMPO can
outperform normal optimization.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 27

Convergence of the GloMPO manager is controlled by :class:`.BaseChecker` objects. These are small classes which define a
single termination condition. These classes can then be easily combined to create sophisticated termination conditions
using :code:`&` and :code:`|` symbolics.

In this case we would like the optimizer to run for a fixed number of iterations or stop if the global minimum is found.
Of course we would not know the global minimum in typical problems but we do in this case.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 29-30

We will configure the optimizers as was done in the :ref:`Minimal` example:

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 32-35

The :ref:`Minimal` example discussed the importance of load balancing. In this example we will override the default
number of slots and limit the manager to 10:

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 37

:class:`.BaseHunter` objects are setup in a similar way to :class:`.BaseChecker` objects and control the conditions in
which optimizers are shutdown by the manager. Each hunter is individually documented :ref:`here <Other Hunters>`.

In this example we will use a hunting set which has proven effective on several problems:

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 39-42

.. note::

   :class:`.BaseHunter` and :class:`.BaseChecker` are evaluated lazily this means that in :code:`x | y`, :code:`y` will
   not be evaluated if :code:`x` is :obj:`True` and in :code:`x & y`, :code:`y` will not be evaluated if :code:`x` is
   :obj:`False`.

:class:`.BaseSelector` objects select which optimizers to start but :class:`.BaseGenerator` objects select a point in
parameter space where to start them.

In this example we will use the :class:`.RandomGenerator` which starts optimizers at random locations.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 44

GloMPO supports running the optimizers both as threads and processes. Processes are preferred and the default
since they circumvent the
`Python Global Interpreter Lock <https://docs.python.org/3.6/glossary.html#term-global-interpreter-lock>`_ but threads
can also be used for tasks that are not multiprocessing safe. In this example we will use processes.

.. attention::

   It is highly recommended that the user familiarize themselves with Python's behavior in this regard! If all
   computations are performed within Python than multithreading will NOT result in the distribution of calculations
   over more than one core.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 46

GloMPO includes a dynamic scope allowing one to watch the optimization progress in real-time using a graphic.
This can be very helpful when configuring GloMPO and the results can be saved as movie files. This functionality
requires `matplotlib <http://matplotlib.sourceforge.net/>`_ and `ffmpeg <ffmpeg.org>`_ installed on the system.

This is turned on for this example but if the script fails simply set :code:`visualisation` to :obj:`False` to
bypass it. Note also that the scope is very helpful for preliminary configuration but is slow due to
`matplotlib <http://matplotlib.sourceforge.net/>`_\\'s limitations and should not be used during production runs.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 48-57

For buggy tasks which are liable to fail or produce extreme results, it is possible that optimizers can get stuck and
simply never return. If this is a risk that we can send a timeout condition after which the manager will force them to
crash. Note that this will not work on threaded backends. In this example this is not needed so we leave the default as
-1.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 59

GloMPO supports checkpointing. This means that its state can be persisted to file during an optimization and this
checkpoint file can be loaded by another GloMPO instance to resume the optimization from that point. Checkpointing
options are configured through a :class:`.CheckpointingControl` instance. In this case we will produce a checkpoint
called :code:`'customized_completed_<DATE>_<TIME>.tar.gz'` once the task has converged.

.. literalinclude:: ../examples/customized.py
   :linenos:
   :lineno-match:
   :lines: 61-63

All arguments are now fed to the manager initialisation. Other settings which did not warrant detailed discussion above
are commented upon below:

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

The :download:`nudging <../examples/nudging.py>` example is a variation of the :ref:`Customized` one. GloMPO will be
run on the same task with virtually the same configuration, but in this case good iterations will be shared between
optimizers. The optimizers, in turn, will use this information to accelerate their convergence. The user should see a
marked improvement in GloMPO's performance. Only two modifications to the :ref:`Customized` example are necessary:

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
