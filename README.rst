
GloMPO
######

============
Introduction
============

GloMPO (**Glo**\bally **M**\anaged **P**\arallel **O**\ptimisation) is a meta-optimizer
which supervises parallel sets of traditional optimization routines using customisable
heuristics. By monitoring the performance of each of these optimizers in real time,
the GloMPO manager is able to make decisions to terminate and start new optimizers in
better locations.

GloMPO is designed to be used on multimodal black-box optimization problems but simpler
problems are not precluded.

The three main advantages to optimization in this way:

1. Optimizers are pushed out of local minima, thus better solutions are more likely
   to be found;

2. Through terminations of optimizers stuck in local minima, the overall computational
   cost of the optimization can be significantly reduced;

3. The use of multiple optimizers allows multiple competitive/equivalent solutions to
   be found.

============
Installation
============

1. Installation is easy after download:

   * Using pip:

     .. code-block:: bash

        cd /path/to/glompo
        pip install .

     If you are developing for GloMPO, you may prefer to install in developer mode:

     .. code-block:: bash

        cd /path/to/glompo
        pip install -e .

   * Using conda:

     .. code-block:: bash

        cd /path/to/glompo
        conda install .

2. Both ``pip`` and ``conda`` will only install core GloMPO dependencies,
   Packages required for optional features must be installed manually. These
   features and their dependencies can be consulted in the ``extra_requires``
   option of ``setup.py``.

3. You should confirm that everything is working correctly by running the tests in the
   ``tests`` folder. Running the tests requires ``pytest`` be installed to your Python
   environment. This is not installed automatically with GloMPO.

   Depending on the abilities of the system core and scope tests can be run together
   or independently:

   .. code-block:: bash

     cd /path/to/glompo
     pytest tests

   NOTE: Tests which require optional components will be automatically skipped if the
   required packages are not installed.

   If your tests fail, please raise an issue on on `GitHub <https://github.com/
   mfgustavo/glompo/issues/new>`_.

=====
Usage
=====

Basic Example
=============

Usage of GloMPO requires, at a minimum, specification of the task to be minimised,
the bounds of the parameters and the local optimizers to be used.

GloMPO includes a set of common multidimensional global optimization test functions
which are useful to benchmark different configurations of the manager. In this example
the Shubert function will be used.

.. code-block:: python

 from glompo import GloMPOManager
 from glompo.opt_selectors import CycleSelector
 from glompo.optimizers import CMAOptimizer  # Requires cma package
 from glompo.benchmark_fnc import Shubert

 task = Shubert()

 manager = GloMPOManager(task=task,
                         opt_selector=CycleSelector([CMAOptimizer]),
                         bounds=task.bounds)

 result = manager.start_manager()

 print(f"Minimum found: {result.fx}")

For a more detailed explanation of GloMPO's use, please consult the ``examples`` folders
and the documentation in ``core/manager.py``

Results
=======

GloMPO produces various types of results files which can be configured via the manager;
all or none of the following can be produced. A summary human-readable YAML file is the
most basic record of the optimization. It includes all GloMPO settings, the final result,
computational resources used, checkpoints created, as well as time and date information.

Image files of the optimizer trajectories can also produced, this requires the `matplotlib`
package and is a helpful way to analyze the optimization performance at a glance.

Finally, all iteration and metadata information from the optimizers themselves is now
saved in a compressed HDF5 format. This is more flexible and user-friendly than the
previous YAML files created by v2 GloMPO. This file also contains all the manager metadata;
in this way all information from an optimization can be accessed from one location. To work
with these files within a Python environment, we recommend loading it with the
`Pytables` module. To explore the file in a user-friendly GUI, we recommend using
the `vitables` package.

Optimization Tasks
=====================

GloMPO is very flexible in terms of the tasks it will accept to minimize.
The task may be a function or an object method and, at the absolute minimum, must
support the following API:

.. code-block:: python

   def __call__(parameter_vector: Sequence[float]) -> float:
       ...

The are some scenarios where the function must return extra information. Either for logging
and later analysis, or for use by the optimizer itself. In that case GloMPO also supports
a more extended task API. The user is directed to the `BaseFunction` class for details.
Note, actual tasks do not need to sub-class this method, it serves only as a template.

Logging
=======

Logging is built into GloMPO and users may optionally configure its logging capability
before running the manager in order to track its progress. Without this manual
configuration the opt_log will not print anywhere! This is mainly used to debug the
code and track execution through the program but it is helpful to send INFO level
messages to the stdout to follow the execution process.

The logging provided in this way is distinct from the summary files provided
at the end of the GloMPO run which are regulated by the summary_files parameter in
the initialisation of ``GloMPOManger``.

The GloMPO logger is called ``glompo`` and components have individual loggers too,
allowing filtering if desired. These are: ``glompo.manager``, ``glompo.checker``,
``glompo.scope``, ``glompo.logger``, ``glompo.generator``, ``glompo.hunter``,
``glompo.selector`` and ``glompo.optimizers``. Logging from optimizers can be
accessed collectively via ``glompo.optimizers`` or individually for each optimizer
via ``glompo.optimizers.optX`` where X is the ID number of the optimizer
(see common/logging.py for a useful Filter which automatically redirects new
optimizers to new log files).

Within user written plug-ins such as custom hunters and convergence criteria, a
``self.logger`` attribute is present and can be used to log behaviour. The
interested user is directed to the Python documentation for the `logging <https:
//docs.python.org/3.9/library/logging.html?#module-logging>`_ package
for details on how to customise this functionality.

An example configuration may look like:

.. code-block:: python

  formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(lineno)d : %(name)s :: %(message)s")

  handler = logging.FileHandler('glompo.opt_log', 'w')
  handler.setFormatter(formatter)

  logger = logging.getLogger('glompo')
  logger.addHandler(handler)
  logger.setLevel('INFO')

  manager = GloMPOManager(...)
  manager.start_manager(...)

Resource Balancing
==================

Resource balancing is critical to GloMPO's success. The typical GloMPO execution
hierarchy takes the following form:

.. image:: _png/hierarchy.png

The first level of parallelization is done at the manager level and controls how the
optimizer routines are spun-off from the manager. This can be done using multiprocessing
or multithreading and is controlled by sending 'processes' or 'threads' to
the `backend` parameter of the GloMPOManager initialisation method. Processes are
preferable to threads as they sidestep Python's Global Interpreter Lock but there are
scenarios where this is inappropriate.

The second level of parallelization is optimizer specific and present in swarm type
optimizers like CMA which require multiple function evaluations per optimizer iteration.
These too can generally be evaluated in parallel using processes or threads.
This can be configured by sending 'processes', 'threads' or 'processes_forced' to the
`backend` parameter of `BaseOptimizer` objects during initialisation (see
``BaseOptimizer`` documentation for details). To avoid crashes (see table below) GloMPO
defaults to threading at this level. Parallelisation at this level is not always
advisable and should only be used in cases where the function evaluation itself is very expensive.

In the case where the function being minimized is in pure python (and there are no
calls to processes outside of python or calculations based on I/O calls) then load
balancing will become challenging due to Python's own limitations:

=========  =========  =====
Parallelization       Setup
--------------------  -----
Level 1    Level 2
=========  =========  =====
Threads    Threads    Total lock within a single Python process due to the GIL. No parallelism can be achieved unless the bulk of the calculation time is spent in an external subprocess.
Threads    Processes  Heavy burden on single process to run the manager and optimizer routines but the load can be adequately distributed over all available resources if the function evaluations are slow enough that the single manager / optimizers process does not become a bottleneck.
Processes  Threads    Not advisable. Processes are launched for each optimizer but parallel function evaluations (which should be more expensive than the optimization routine itself) is threaded to no benefit due to the GIL.
Processes  Processes  Theoretically the ideal scenario which guarantees perfect parallelism and full use of available resources. However, Python does not allow daemonic processes (optimizers) to spawn children (parallel function evaluations). Turning off daemonic spawning of optimizers is risky as it is possible they will not be cleaned-up if the manager crashes. GloMPO does, however, do its best to deal with this eventuality but there are scenarios where children are not collected.
=========  =========  =====

.. note::
   We emphasize here that these difficulties only arise when attempting to load balance
   over two parallelization levels.

As explained in the above table achieving process parallelism at both levels is not
straightforward but GloMPO does support an avenue to do this, however, its use is
**not recommended**: the user may send `'processes_forced'` to the `backend` parameter
of the GloMPO manager initialisation. This will spawn optimizers non-daemonically.

 .. warning::
    This method is **not recommended**. It is unsafe to spawn non-daemonic
    processes since these expensive routines will not be shutdown if the manager
    were to crash. The user would have to terminate them manually.

Manual Control
==============

GloMPO supports manual control of optimizer termination. The user may create stop
files in the working directory which, when detected by the manager, will shutdown
the chosen optimizer.

Files must be called `STOP_x` where `x` is the optimizer ID number. This file name
is case-sensitive. Examples include `STOP_1` or `STOP_003`. Note that these files
should be empty as they are deleted by the manager once processed.

Execution Information
=====================

GloMPO logs include information about CPU usage, memory usage and system load. This
is useful traceback to ensure the function is being parallelized correctly. It is
important to note that CPU usage and memory usage is provided at a *process level*
system load is provided at a *system level*. This means that the system load
information will only be of use if GloMPO is the only application running over the
entire system. In distributed computing systems where GloMPO is only given access to
a portion of a node, this information will be useless as it will be conflated with
the usage of other users.

Checkpointing
=============

Checkpointing tries to create an entire image of the GloMPO state, it is the user's
responsibility to ensure that the used optimizers are restartable. Within
`tests/test_optimizers.py` there is the `TestSubclassesGlompoCompatible` class which
can be used to ensure an optimizer is compatible with all of GloMPO's functionality.

The optimization task can sometimes not be reduced to a pickled state depending on
it complexity and interfaces to other codes. GloMPO will first attempt to `pickle`
the object, failing that GloMPO will attempt to call the `checkpoint_save()` function if
the task has such a method. If this also fails the checkpoint is created without the
optimization task. GloMPO can be restarted from an incomplete checkpoint if the
missing components are provided.

Similarly to manual stopping of optimizers, manual checkpoints can also be requested
by created a file named `CHKPT` in the working directory. Note, that this file will
be deleted by the manager when the checkpoint is created.
