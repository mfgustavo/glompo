=======
Hunters
=======

Abstract Base Hunter
====================

.. autoclass:: glompo.hunters.basehunter.BaseHunter
   :members:
   :special-members: __str__
   :inherited-members: str_with_result

Combining Base Hunters
======================

:class:`.BaseHunter` is based on the same structure as :class:`.BaseChecker`. Thus, simple conditions can also be
combined into more sophisticated termination conditions. See :ref:`Combining Base Checkers`.

Included Hunters
=================

For convenience, GloMPO comes bundled with several simple hunters already included.

.. automodule:: glompo.hunters
   :members:
   :show-inheritance:
   :exclude-members: BaseHunter
