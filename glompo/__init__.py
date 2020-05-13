

""" GloMPO (Globally Managed Parallel Optimization)
    Author: Michael Freitas Gustavo

    GloMPO is designed to run several optimization algorithms in parallel on a given minimisation task and monitor
    the performance of each. Based on this, the system can terminate poor performing optimizers early and start new
    ones in promising locations.

    ---
    Notes:
        Logging is built into GloMPO and users may optional configure its logging capability before running the
        manager in order to track its output. Without this manual configuration the opt_log will not print anywhere!
        This is mainly used to debug the code and track execution through the program. It is helpful to send INFO level
        messages to the stdout to follow the execution process.

        The logging provided in this way is distinct from the summary opt_log-file provided at the end of the GloMPO run
        and regulated by the summary_files parameter in the __init__ method.

        The GloMPO logger is called 'glompo' and components have individual loggers too, allowing filtering if
        desired. These are: 'glompo.checker', 'glompo.scope', 'glompo.logger', 'glompo.generator', 'glompo.hunter',
        'glompo.selector' and 'glompo.optimizers'. Logging from optimizers can be accessed collectively via
        'glompo.optimizers' or individually for each optimizer via 'glompo.optimizers.optX' where X is the ID number
        of the optimizer (see common/logging.py for a useful Filter which automatically redirects new optimizers to
        new streams).

        Within user written plug-ins such as custom hunters and convergence criteria, a self.logger attribute is
        present and can be used to log behaviour.

        An example configuration may look like:
            formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(lineno)d : %(name)s :: %(message)s")

            handler = logging.FileHandler('glompo.opt_log', 'w')
            handler.setFormatter(formatter)

            logger = logging.getLogger('glompo')
            logger.addHandler(handler)
            logger.setLevel('INFO')

            manager = GloMPOManager(...)
            manager.start_manager(...)
"""

import logging

from glompo.core.manager import GloMPOManager


__all__ = ("GloMPOManager",)


logging.getLogger('glompo').addHandler(logging.NullHandler())
