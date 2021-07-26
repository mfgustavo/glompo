=====
Tasks
=====

General Optimization Tasks
==========================

In general, GloMPO is very flexible in terms of the optimization functions it accepts. Thus, both methods and functions
may be used. At the absolute minimum GloMPO expects a single numerical value to be returned when the task is called with
a vector of numbers from within the bounded parameter space. GloMPO does, however, support extended functionality.

The :class:`BaseFunction <glompo.core.function.BaseFunction>` class provided in the package, and detailed below, serves
as an API guide to what is expected and possible.

.. autoclass:: glompo.core.function.BaseFunction
   :members:

Benchmark Test Functions
========================

For convenience, a collection of benchmark functions is bundled with GloMPO. These may be helpful for testing purposes
and may be used to experiment with different configurations and ensure a script is functional before being applied to a
more expensive test case.

.. py:currentmodule:: glompo.benchmark_fncs

.. autosummary::
   :nosignatures:

   Ackley
   Alpine01
   Alpine02
   Deceptive
   Easom
   ExpLeastSquaresCost
   Griewank
   Langermann
   Levy
   Michalewicz
   Qing
   Rana
   Rastrigin
   Rosenbrock
   Schwefel
   Shekel
   Shubert
   Stochastic
   StyblinskiTang
   Trigonometric
   Vincent
   ZeroSum

.. automodule:: glompo.benchmark_fncs
   :member-order: alphabetical
   :ignore-module-all:
   :members:
   :special-members: __init__
   :show-inheritance:
   :exclude-members: BaseTestCase, bounds, delay, dims, min_fx, min_x

   .. autoclass::  glompo.benchmark_fncs.BaseTestCase
      :members:
