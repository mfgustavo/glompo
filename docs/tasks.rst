=====
Tasks
=====

General Optimization Tasks
==========================

In general, GloMPO is very flexible in terms of the optimization functions it accepts. Thus, both methods and functions
may be used. At the absolute minimum GloMPO expects a single numerical value to be returned when the task is called with
a vector of numbers from within the bounded parameter space. GloMPO does, however, support extended functionality.

The :class:`BaseFunction <glompo.core.function.BaseFunction>` class provided in the package and detailed below serves as
an API guide to what is expected and possible.

.. autoclass:: glompo.core.function.BaseFunction
   :members:

Benchmark Test Functions
========================

.. automodule:: glompo.benchmark_fncs
   :member-order: alphabetical
   :members:
   :show-inheritance:
   :exclude-members: BaseTestCase, bounds, delay, dims, min_fx, min_x

   .. autoclass::  glompo.benchmark_fncs._base.BaseTestCase
      :members:
