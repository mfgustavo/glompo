
GloMPO
######

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

3.Tests are available in the ``tests`` folder and split into core tests and plotting
  related tests which are optional features. Depending on the abilities of the system
  these can be run together or independently:

  .. code-block:: bash

     cd glompo/glompo
     pytest tests
     # or
     pytest tests/core_tests
     pytest tests/scope_tests

  NOTE: core_tests will require the extra package dependencies (except `matplotlib`)
  discussed in point 2.
