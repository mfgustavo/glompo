

""" GloMPO (Globally Managed Parallel Optimization)
    Author: Michael Freitas Gustavo

    GloMPO is designed to run several optimization algorithms in parallel on a given minimisation task and monitor
    the performance of each. Based on this, the system can terminate poor performing optimizers early and start new
    ones in promising locations.
"""

import logging

from glompo.core.manager import GloMPOManager


__all__ = ("GloMPOManager",)


logging.getLogger('glompo').addHandler(logging.NullHandler())
