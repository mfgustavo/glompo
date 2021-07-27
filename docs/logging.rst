.. _Logging Messages:

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

The GloMPO logger is called ``glompo`` and components have individual loggers too,
allowing filtering if desired. These are: ``glompo.manager``, ``glompo.checker``,
``glompo.scope``, ``glompo.logger``, ``glompo.generator``, ``glompo.hunter``,
``glompo.selector`` and ``glompo.optimizers``. Logging from optimizers can be
accessed collectively via ``glompo.optimizers`` or individually for each optimizer
via ``glompo.optimizers.optX`` where X is the ID number of the optimizer
(see common/logging.py for a useful Filter which automatically redirects new
optimizers to new log files).

Within user written plug-ins such as custom hunters and convergence criteria, a
``self.logger`` attribute is present and can be used to log behaviour. The
interested user is directed to the Python documentation for the `logging <https:
//docs.python.org/3.9/library/logging.html?#module-logging>`_ package
for details on how to customise this functionality.

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

Note that status messages are delivered
through `logging.INFO <https://docs.python.org/3.6/library/logging.html?highlight=logging%20info#logging-levels>`_
level message. Logging must be enabled and setup to see these messages.
See :ref:`Logging Messages` for more information.

Resource Usage Logging
**********************

GloMPO logs include information about CPU usage, memory usage and system load. This
is useful traceback to ensure the function is being parallelized correctly. It is
important to note that CPU usage and memory usage is provided at a *process level*
system load is provided at a *system level*. This means that the system load
information will only be of use if GloMPO is the only application running over the
entire system. In distributed computing systems where GloMPO is only given access to
a portion of a node, this information will be useless as it will be conflated with
the usage of other users.