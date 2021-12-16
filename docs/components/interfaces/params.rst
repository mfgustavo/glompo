ParAMS
======

.. toctree::
   :maxdepth: 4
   :caption: Table of Contents

   params/glompoopt
   params/errorfncs
   params/optimization
   params/paramsbuilders

`ParAMS <https://www.scm.com/doc/params/index.html>`_ is a reparameterization tool for computational chemistry models which ships with SCM's AMS suite. Several interfaces to ParAMS have been included in the GloMPO package. They allow GloMPO to manage ReaxFF and GFN-xTB reparameterisations.

There are several ways to interface the two pieces of software, depending on your preferred workflow or interface. All the classes and methods can be located in the submodules of :mod:`!glompo.interfaces.params`.

1. The first interface method wraps GloMPO to look like a :class:`scm.params.optimizers.base.BaseOptimizer`. This is then passed to a :class:`~scm.params.core.parameteroptimization.Optimization` instance as normal. This is the most comfortable method for those used to ParAMS. See :ref:`GloMPO as ParAMS Optimizer`.

2. For those who would like more control of the optimization cost function (for example, to be able to perform sensitivity analysis), or would like to work primarily through GloMPO interfaces, they can use the error function classes in :ref:`ParAMS as Error Functions`. These classes form the :attr:`.GloMPOManager.task`.

  Several convenience function are also available :ref:`here <Builder Functions>` to help setup the necessary components for these classes.

3. In time, GloMPO will become the default ParAMS optimizer. For this a modified version of :class:`~scm.params.core.parameteroptimization.Optimization` is available :ref:`here <GloMPO as Default ParAMS Optimizer>`.

   .. attention::

      This interface requires ParAMS > v0.5.1

   .. warning::

      This interface option is in early stage development and is likely to undergo significant changes as both GloMPO and ParAMS API is changed to interface the codes together.
