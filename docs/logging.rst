****************
Logging Messages
****************

Logging is built into GloMPO and users may optionally configure its logging capability before running the manager in
order to track its progress. Without this manual configuration no progress or status message will print at all! The
logging system can be used to debug the code, but is most helpful in tracking execution through the program by sending
`logging.INFO <https://docs.python.org/3.6/library/logging.html?highlight=logging%20info#logging-levels>`_ level
messages and above to :obj:`python:sys.stdout`.

The logging provided in this way is distinct from the summary files provided at the end of the GloMPO run which are
regulated by :attr:`.GloMPOManager.summary_files`.

The GloMPO logger is called :code:`'glompo'` and components have individual loggers too, allowing filtering if desired.
These are: :code:`'glompo.manager'`, :code:`'glompo.checker'`, :code:`'glompo.scope'`, :code:`'glompo.logger'`,
:code:`'glompo.generator'`, :code:`'glompo.hunter'`, :code:`'glompo.selector'` and :code:`'glompo.optimizers'`. Logging
from optimizers can be accessed collectively via :code:`'glompo.optimizers'` or individually for each optimizer via
:code:`'glompo.optimizers.optX'` where :code:`'X'` is the ID number of the optimizer (see :class:`.SplitOptimizerLogs`
for the :class:`logging.Filter` needed to achieve this).

Within user written plug-ins, such as custom hunters and convergence criteria, a :attr:`!logger` attribute (instance of
:class:`logging.Logger`) is present and can be used to log behaviour. The interested user is directed to the Python
documentation for the :mod:`logging` package for details on how to customise this functionality.

An example configuration may look like:

.. code-block:: python

  formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(lineno)d : %(name)s :: %(message)s")

  handler = logging.FileHandler('glompo.opt_log', 'w')
  handler.setFormatter(formatter)

  logger = logging.getLogger('glompo')
  logger.addHandler(handler)
  logger.setLevel('INFO')

  manager = GloMPOManager(...)
  manager.start_manager(...)
