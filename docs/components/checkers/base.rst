BaseChecker
===========

A collection of subclasses of
:class:`~glompo.convergence.basechecker.BaseChecker` are provided, these can be used in combinations of and
(:code:`&`) and or (:code:`|`) to tailor various conditions. For example::

   convergence_criteria = MaxFuncCalls(20000) | KillsAfterConvergence(3, 1) & MaxSeconds(60*5)

In this case GloMPO will run until 20 000 function evaluations OR until 3 optimizers have been killed
after the first convergence provided it has at least run for five minutes.

Default: :class:`KillsAfterConvergence(0, 1) <glompo.convergence.nkillsafterconv.KillsAfterConvergence>`
i.e. GloMPO terminates as soon as any optimizer converges.

.. note::

   For performance and to allow conditionality, conditions are evaluated 'lazily' i.e. :code:`x or y` will return if `x`
   is :obj:`True` without evaluating `y`. :code:`x and y` will return :obj:`False` if `x` is :obj:`False` without
   evaluating `y`.

.. autoclass:: glompo.convergence.basechecker.BaseChecker
   :members:
