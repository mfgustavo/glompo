ParAMS
======

`ParAMS <https://www.scm.com/doc/params/index.html>`_ is a reparameterization tool for computational chemistry models which ships with SCM's AMS suite. Several interfaces to ParAMS have been included in the GloMPO package. They allow GloMPO to manage ReaxFF and GFN-xTB reparameterisations.

There are two ways to interface the two pieces of software, depending on your preferred workflow or interface:

#. ParAMS is primary, setup an :class:`~scm.params.core.parameteroptimization.Optimization` instance as normal. GloMPO is wrapped using the :class:`.GlompoParamsWrapper` to look like a :class:`scm.params.optimizers.base.BaseOptimizer`.

#. GloMPO is primary, setup a :class:`.GloMPOManager` instance as normal. The :class:`.ReaxFFError` class below will create the error function to be used as the manager :attr:`.GloMPOManager.task`.

The second approach is recommended.

.. automodule:: glompo.interfaces.params
   :members:
   :show-inheritance:
   :ignore-module-all:
