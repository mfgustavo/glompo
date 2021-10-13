========
Checkers
========

Abstract Base Checker
=====================

.. autoclass:: glompo.convergence.basechecker.BaseChecker
   :members:
   :special-members: __str__
   :inherited-members: str_with_result

Combining Base Checkers
=======================

Instances of :class:`.BaseChecker` can be combined with :code:`&` and :code:`|` boolean operations. This allows individual checkers to be very simple, but be combined into more sophisticated conditions. Combining checkers in this way produces a new :class:`.BaseChecker` object which itself can be further combined. This allows conditions to be as deeply nested as one desires. For example:

   >>> a = CheckerA()
   >>> b = CheckerB()
   >>> c = CheckerC()
   >>> d = CheckerD()
   >>> combi = a & b | c & d

The order of evaluation is :code:`&` before :code:`|`, thus the above would be equivalent to:

   >>> combi = ((a & b) | (c & d))

.. important::

   A combined :class:`.BaseChecker` is evaluated lazily. This means:

      #. :code:`a & b` will not evaluate :code:`b` if :code:`a` is :obj:`False`
      #. :code:`a | b` will not evaluate :code:`b` if :code:`a` is :obj:`True`

Lazy evaluation ensures a faster return, and explains the presence of :obj:`None` when a hunt evaluation statement is printed. For example:

   >>> combi(...)
   True
   >>> from glompo.common.helpers import nested_string_formatting
   >>> nested_string_formatting(combi.str_with_result())
   '[
     CheckerA() = True &
     CheckerB() = True
    ] |
    [
     CheckerC() = None &
     CheckerD() = None
    ]'

Make sure to structure your nesting in the correct order. For example, if you want to make sure a certain checker is always evaluated, place it first. If a checker is slow to evaluate, place it last.

All of the above holds true for :ref:`Hunters` too as they share a common hidden base.

Included Checkers
=================

For convenience, GloMPO comes bundled with several simple checkers already included.

.. automodule:: glompo.convergence
   :members:
   :show-inheritance:
   :exclude-members: BaseChecker
