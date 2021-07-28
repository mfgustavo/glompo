=========
Selectors
=========

Abstract Base Selector
======================

.. autoclass:: glompo.opt_selectors.baseselector.BaseSelector
   :members:
   :special-members: __contains__, __init__

Spawn Control
=============

As seen above, spawning new optimizers can be stopped without shutting down the entire optimization. This can be used to
make sure that functioning optimizers are given time to explore, and function evaluations are not wasted on new
optimizers towards the end of an optimization when they will be unlikely to have enough time to develop. GloMPO comes
bundled with a couple of simple convenient controllers.

The :meth:`__call__` method of these classes is expected to be implemented as follows:

.. method:: __call__(manager: GloMPOManager)

   Evaluated everytime :meth:`.BaseSelector.select_optimizer` is called. Determines if new optimizers should stop
   spawning for the remainder of the optimization.

   :return: :obj:`True` if spawning should be allowed, :obj:`False` otherwise.
   :rtype: bool

.. automodule:: glompo.opt_selectors.spawncontrol
   :members:

Included Selectors
==================

For convenience, GloMPO comes bundled with several simple selectors already included.

.. automodule:: glompo.opt_selectors
   :members:
   :show-inheritance:
   :exclude-members: BaseSelector, select_optimizer
