==========
Generators
==========

Abstract Base Generator
=======================

.. autoclass:: glompo.generators.basegenerator.BaseGenerator
   :members:

Simple Generators
=================

For convenience, GloMPO comes bundled with several simple generators already included.

.. automodule:: glompo.generators
   :members:
   :show-inheritance:
   :exclude-members: BaseGenerator, generate

Advanced Generators
===================

GloMPO also comes bundled with two more advanced sampling strategies.

.. autoclass:: glompo.generators.annealing.AnnealingGenerator

.. autoclass:: glompo.generators.basinhopping.BasinHoppingGenerator
