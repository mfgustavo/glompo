
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

GloMPO is currently hosted on `Github <https://github.com/mfgustavo/glompo.git>`_ as
a private repository. Contact michael.freitasgustavo@ugent.be for access.

1. Installation is easy after download:

   * Using pip:

     .. code-block:: bash

        cd /path/to/glompo
        pip install .

   * Using conda:

     .. code-block:: bash

        cd /path/to/glompo
        conda install .

2. Both ``pip`` and ``conda`` will only install core GloMPO dependencies,
   Packages required for optional features must be installed manually. These
   features and their dependencies can be consulted in the ``extra_requires``
   option of ``setup.py``.

3. Tests are available in the ``tests`` folder and split into core tests and plotting
   related tests which are optional features. Running the tests requires ``pytest``
   be installed to your Python environment. This is not installed automatically with
   GloMPO.

   Depending on the abilities of the system core and scope tests can be run together
   or independently:

   .. code-block:: bash

     cd glompo/glompo
     pytest tests
     # or
     pytest tests/core_tests
     pytest tests/scope_tests

   NOTE: core_tests will only test optional components if the required packages are
   installed.

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

For a more sophisticated setup with specially configured hunting and convergence
components consult the documentation in ``core/manager.py``

Logging
=======

Logging is built into GloMPO and users may optional configure its logging capability
before running the manager in order to track its progress. Without this manual
configuration the opt_log will not print anywhere! This is mainly used to debug the
code and track execution through the program but it is helpful to send INFO level
messages to the stdout to follow the execution process.

The logging provided in this way is distinct from the summary opt_log-file provided
at the end of the GloMPO run and regulated by the summary_files parameter in
the ``__init__`` method.

The GloMPO logger is called ``glompo`` and components have individual loggers too,
allowing filtering if desired. These are: ``glompo.manager``, ``glompo.checker``,
``glompo.scope``, ``glompo.logger``, ``glompo.generator``, ``glompo.hunter``,
``glompo.selector`` and ``glompo.optimizers``. Logging from optimizers can be
accessed collectively via ``glompo.optimizers`` or individually for each optimizer
via ``glompo.optimizers.optX`` where X is the ID number of the optimizer
(see common/logging.py for a useful Filter which automatically redirects new
optimizers to new log files).

Within user written plug-ins such as custom hunters and convergence criteria, a
``self.logger`` attribute is present and can be used to log behaviour.

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