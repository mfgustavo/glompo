.. Add banner here

.. image:: docs/_static/banner.gif
   :width: 800

..  Add buttons here
|
.. image:: https://img.shields.io/github/v/tag/mfgustavo/glompo?style=for-the-badge
.. image:: https://img.shields.io/github/last-commit/mfgustavo/glompo?style=for-the-badge
.. image:: https://img.shields.io/github/issues-raw/mfgustavo/glompo?style=for-the-badge
.. image:: https://img.shields.io/github/license/mfgustavo/glompo?style=for-the-badge

.. Describe your project in brief

.. describe-start

GloMPO (**Glo**\bally **M**\anaged **P**\arallel **O**\ptimization) is an optimisation framework which supervises and
controls traditional optimization routines in real-time using customisable heuristics. By monitoring the performance of
each of these optimizers in real time, the GloMPO manager is able to make decisions to terminate and start new
optimizers in better locations.

GloMPO is designed to be used on high-dimensional, expensive, multimodal, black-box optimization problems but simpler
problems are not precluded.

The three main advantages to optimization in this way:

1. Optimizers are pushed out of local minima, thus more and better solutions are more likely to be found;

2. Through terminations of optimizers stuck in local minima, function evaluations can be used more efficiently;

3. The use of multiple optimizers allows multiple competitive/equivalent solutions to be found.

.. describe-end

.. image:: docs/_static/demo.gif
   :width: 500
   :align: center

.. _Back to Top:

.. contents:: Table of Contents
   :local:
   :depth: 2

############
Installation
############

[`Back to Top`_]

.. install-start

The source code may be downloaded directly from `GitHub <https://github.com/mfgustavo/glompo>`_, or it may be cloned
into a target directory using:

.. code-block:: bash

    git clone https://github.com/mfgustavo/glompo.git

Installation is easy after download:

.. code-block:: bash

    cd /path/to/glompo
    pip install .

This will copy the GloMPO source code into your Python environment. If you are developing for GloMPO, you may prefer to
install in developer mode:

.. code-block:: bash

    cd /path/to/glompo
    pip install -e .

This will not copy the source code and GloMPO will be read directly from the directory into which it was downloaded or
extracted.

.. install-end

The installation will only install core GloMPO dependencies. Packages required for optional features must be installed
manually. These features and their dependencies can be consulted in the
`documentation <https://glompo.readthedocs.io/installation.html>`_.

To install GloMPO with optional dependencies:

.. code-block:: bash

    pip install .[cma,checkpointing,...]

#####
Tests
#####

.. test-start

You should confirm that everything is working correctly by running the tests in the ``tests`` folder. Running the tests
requires ``pytest`` be installed to your Python environment. This is not installed automatically with GloMPO, but can be
done with the ``testing`` install option.

.. code-block:: bash

   cd /path/to/glompo
   pytest

.. note::
    Tests which require optional components will be automatically skipped if the required packages are not installed.

.. test-end

.. note::
    If your tests fail, please raise an issue as detailed in the `Issues`_ section.



#####
Usage
#####

Basic Example
#############

[`Back to Top`_]

Usage of GloMPO requires, at a minimum:

#. Specification of the task to be minimised;

#. The bounds of the parameters;

#. The local optimizers to be used.

GloMPO includes a set of common multidimensional global optimization test functions
which are useful to benchmark different configurations of the manager. In this example
the Shubert function will be used.

.. code-block:: python

   from glompo.core.manager import GloMPOManager
   from glompo.opt_selectors import CycleSelector
   from glompo.optimizers import CMAOptimizer  # Requires cma package
   from glompo.benchmark_fnc import Shubert

   task = Shubert()

   manager = GloMPOManager(task=task,
                           opt_selector=CycleSelector([CMAOptimizer]),
                           bounds=task.bounds)

   result = manager.start_manager()

   print(f"Minimum found: {result.fx}")

For a more detailed explanation of GloMPO's use, please consult the ``examples`` folder and the `documentation <unknown>`_.

Structure
#########

Below is a brief introduction to the most important components of the code to orientate first-time users. GloMPO is
implemented in a modular way such that all decision criteria is customizable.

``core``
   This package contains the most important GloMPO components:

   ``manager.py``
        Contains ``GloMPOManager`` the primary point of entry into the code. The manager performs the actual
        optimzation, accepts all settings, and produces all the output.

   ``checkpointing.py``
        Contains ``CheckpointingControl`` which configures GloMPO's ability to save a snapshot of itself during an
        optimization from which it can resume later.

   ``function.py``
        An API template for the optimization task from which it *may*, but *need not*, inherit.

   ``scope.py``
        GloMPO infrastructure to produce real-time video recordings of optimizations.

``opt_selectors``
   Each file contains a different ``BaseSelector`` child-class. These objects decide which optimizer configuration to
   start from a list of options.

``optimizers``
   Each file contains a different ``BaseOptimizer`` child-class. These are implementations or wrappers around actual
   optimization algorithms.

``generators``
   Each file contains a different ``BaseGenerator`` child-class. These are algorithms which decide where optimizers are
   started within the search domain.

``convergence``
   Each file contains a different ``BaseChecker`` child-class. These are simple conditions which control GloMPO's
   overall termination conditions. These classes/conditions can be combined into more sophisticated ones, for example:

   .. code-block:: python

      MaxSeconds(6000) | MaxFuncCalls(30_000) & MaxOptStarted(5)

``hunters``
   Each file contains a different ``BaseHunter`` child-class. These are termination conditions which, if satisfied,
   will get GloMPO to trigger an early termination of a particular optimizer. These classes/conditions can be combined
   similarly to ``BaseChecker``\s:

``benchmark_fncs``
   A collection of well-known global optimization test functions. These are often faster to evaluate than the actual
   function one wishes to minimize. Using these can be helpful to quickly configure GloMPO before applying it to more
   time-consuming tasks.

######
Issues
######

[`Back to Top`_]

.. issue-start

Raise any issues encountered on the appropriate `GitHub <https://github.com/mfgustavo/glompo/issues/new>`_ page. Please
include a MWE of the problem, a list of packages installed in your python environment, and a detailed description of the
workflow which led to the error.

.. issue-end

#############
Contributions
#############

[`Back to Top`_]

.. contri-start

Contributions are welcome and can be submitted as pull requests `here <https://github.com/mfgustavo/glompo/pulls>`_.
Before contributing new material, please raise a new `issue <https://github.com/mfgustavo/glompo/issues/new>`_ and tag
it as ``enhancement``. This will provide an opportunity to discuss the proposed changes with other contributors before
a new feature is introduced.

Pull request checklist:

#. Please ensure that your contributions follow general :pep:`8` style guidelines;

#. Only submit documented code;

#. Make sure that all existing tests still pass or update the failing ones if they are no longer relevant;

#. Include new tests if the current suite does not cover your contributions;

#. Keep each pull request small and linked to a single issue.

.. contri-end

#######
License
#######

[`Back to Top`_]

.. license-start

GloMPO is licensed under `GPL-3.0 <https://opensource.org/licenses/GPL-3.0>`_.

.. license-end

########
Citation
########

.. citation-start

If you find GloMPO useful, please consider citing the follow article in your work:

.. citation-end