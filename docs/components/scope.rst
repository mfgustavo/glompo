GloMPO Scope
============

GloMPO provides a dynamic plotting object to track optimizer progress. This can be viewed in real-time, or saved into a
movie file. This is useful for debugging and finding the right manager configurations. It provides insight into how the
optimization behaved, and how the manager managed it.

The class details below are for reference only. The user need not initialise or control the scope directly; this is all
done by GloMPO internals. To dynamically plot an optimization, see :attr:`.GloMPOManager.visualisation`.

.. autoclass:: glompo.core.scope.GloMPOScope
   :members:
   :member-order: bysource
