BaseHunter
==========

A collection of subclasses of
:class:`.BaseHunter` are provided, these can be used in combinations of and (:code:`&`) and or (:code:`|`) to tailor
various conditions. For example::

   killing_conditions = (BestUnmoving(100, 0.01) & TimeAnnealing(2) & ValueAnnealing()) | ParameterDistance(0.1)

In this case GloMPO will only allow a hunt to terminate an optimizer if:

#. an optimizer's best value has not improved by more than 1% in 100 function calls,

#. and it fails an annealing type test based on how many iterations it has run,

#. and if fails an annealing type test based on how far the victim's value is from the best optimizer's
   best value,

#. or the two optimizers are iterating very close to one another in parameter space.

Default is :obj:`None` i.e. Killing is not used and the manager will not terminate optimizers.

.. note:

   For performance and to allow conditionality, conditions are evaluated 'lazily' i.e.
   :obj:`x or y` will return if `x` is :obj:`True` without evaluating `y`. :obj:`x and y` will return
   :obj:`False` if `x` is :obj:`False` without evaluating `y`.

.. autoclass:: glompo.hunters.basehunter.BaseHunter
   :members:
   :special-members: __str__
   :inherited-members: str_with_result
