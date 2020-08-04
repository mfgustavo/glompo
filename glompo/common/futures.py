"""
Wrappers for parallelization

This makes it possible to write generic parallel code, either using:

- concurrent.futures.ThreadPoolExecutor
- concurrent.futures.ProcessPoolExecutor
- Scoop (if present)

These are all PEP 3148 compatible but the API proposed in this PEP makes it
hard to write generic parallel code that allows swapping parallelization
libraries easily. The Executor classes defined in this module contain all
functionality inside the class, such that the parallel code only needs to
interact with an instance of the executor, not the package in which it is
defined.

The Executor API from concurrent.futures is followed, with two extra methods:

- wait
- as_completed

In concurrent.futures, these are implemented as functions instead of Executor
methods, making it impractical to switch to different implementation. By
turning them into methods, this becomes easier. Similarly, the following
constants are added as class attributes:

FIRST_COMPLETED
FIRST_EXCEPTION
ALL_COMPLETED

"""

import concurrent.futures as cf
from abc import ABC, abstractmethod

try:
    import scoop.futures as sf

    scoop_present = True
except ImportError:
    scoop_present = False

__all__ = ("ThreadPoolExecutor",
           "ProcessPoolExecutor",
           "ScoopExecutor")


class BaseExecutor(ABC):

    @property
    @abstractmethod
    def _backend(self):
        return None

    @property
    def FIRST_COMPLETED(self):
        return self._backend.FIRST_COMPLETED

    @property
    def FIRST_EXCEPTION(self):
        return self._backend.FIRST_EXCEPTION

    @property
    def ALL_COMPLETED(self):
        return self._backend.ALL_COMPLETED

    def wait(self, fs, timeout=None, return_when=ALL_COMPLETED):
        return self._backend.wait(fs, timeout, return_when)

    def as_completed(self, fs, timeout=None):
        return self._backend.as_completed(fs, timeout)


class _ConcurrentExecutor(BaseExecutor):
    """Wrapper for any executor defined in the standard library concurent.futures.

    If you already have an executor instance from this library, you directly create
    an instance of this base class. Otherwise, it may be more convenient to
    use the ThreadPoolExecutor or ProcessPoolExecutor classes below.

    """

    @property
    def _backend(self):
        return cf


class ThreadPoolExecutor(_ConcurrentExecutor, cf.ThreadPoolExecutor):
    """A Wrapper for cf.ThreadPoolExecutor"""


class ProcessPoolExecutor(_ConcurrentExecutor, cf.ProcessPoolExecutor):
    """A Wrapper for cf.ProcessPoolExecutor"""


if scoop_present:
    class ScoopExecutor(BaseExecutor):
        """Wrapper for scoop in Executor API

        Not subclassing from concurrent.futures.Executor because it does not provide
        much useful functionality.

        """

        @property
        def _backend(self):
            return sf

        @staticmethod
        def submit(fn, *args, **kwargs):
            return sf.submit(fn, *args, **kwargs)

        @staticmethod
        def map(fn, *iterables, timeout=None, chunksize=1):
            # chunksize is ignored.
            return sf.map(fn, *iterables, timeout)

        @staticmethod
        def shutdown(wait=True):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.shutdown(wait=True)
            return False
