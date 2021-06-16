""" Contains code to support using both multiprocessing and threading within the GloMPO manager. """
import logging
import queue
import sys
import threading
import traceback
import warnings
from pathlib import Path
from typing import TextIO

__all__ = ("CustomThread",
           "ThreadPrintRedirect",
           "ChunkingQueue")


class CustomThread(threading.Thread):
    """ Adds an exitcode property to the Thread base class as well as code to redirect the thread printstream to a
        file if this has been setup before hand.
    """

    def __init__(self, working_directory: Path, *args, redirect_print: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exitcode = 0
        self.redirect = redirect_print
        self.working_directory = working_directory

    def run(self):
        try:
            if self.redirect:
                try:
                    opt_id = int(self.name.replace("Opt", ""))
                    sys.stdout.register(opt_id, self.working_directory, 'out')
                    sys.stderr.register(opt_id, self.working_directory, 'err')
                except AttributeError:
                    warnings.warn("Prinstream redirect failed. Print stream will only redirect if ThreadPrintRedirect "
                                  "is setup beforehand.", RuntimeWarning)
            super().run()
        except Exception as e:
            sys.stderr.write("".join(traceback.TracebackException.from_exception(e).format()))
            self.exitcode = -1
            raise e
        finally:
            if self.redirect:
                try:
                    sys.stdout.close(threading.currentThread().ident)
                    sys.stderr.close(threading.currentThread().ident)
                except Exception as e:
                    warnings.warn(f"Closing printstream files failed. Caught exception: {e}", RuntimeWarning)


# Adapted from https://stackoverflow.com/questions/14890997/redirect-stdout-to-a-file-only-for-a-specific-thread
# Author: Golgot
class ThreadPrintRedirect:
    """ Redirects individual threads to their own print stream.

        Notes
        -----
        To implement use the following statements before threads are created:
            sys.stdout = ThreadPrintRedirect(sys.stdout)
            sys.stderr = ThreadPrintRedirect(sys.stderr)

        By default all threads will continue to print to sys.stdout.

        To redirect a thread make it run register as its first command when it is started. This is done automatically
        if CustomThread is used.
    """

    def __init__(self, intercept: TextIO):
        self.stdout = intercept
        self.threads = {}  # Dict[thread_id: int, file: _io.TextIOWrapper]

    def register(self, opt_id: int, working_directory: Path, ext: str):
        """ Adds a thread to the set of files. """
        thread_id = threading.currentThread().ident
        path = working_directory / "glompo_optimizer_printstreams" / f"printstream_{opt_id:04}.{ext}"
        self.threads[thread_id] = path.open("w+")

    def write(self, message):
        """ Sends message to the appropriate handler. """
        ident = threading.currentThread().ident
        if ident in self.threads and not self.threads[ident].closed:
            self.threads[ident].write(message)
        else:
            self.stdout.write(message)

    def flush(self):
        """ Required for Python 3 compatibility. """

    def close(self, thread_id: int = None):
        """ Closes all open files to which messages are being sent. """
        if thread_id:
            self.threads[thread_id].close()
        else:
            for file in self.threads.values():
                file.close()


class ChunkingQueue(queue.Queue):
    """ queue.get calls by the manager can become the performance bottleneck when managing the optimization of very fast
        functions. ChunkedQueue detects when this is the case and begins pushing the results in chunks of several items
        rather than individually.
    """

    def __init__(self, max_queue_size: int = 0, max_chunk_size: int = 1):
        """ Extends functionality of queue.Queue with a chunking system.

            Parameters
            ----------
            max_queue_size: int
                Maximum number of items allowed in the queue at one time.
            max_chunk_size: int
                Number of items grouped together into the cache before being put in the queue.
        """
        super().__init__(max_queue_size)
        self.chunk_size = max_chunk_size
        self.fast_func = False
        self.cache = []
        self.cache_lock = threading.Lock()
        self.logger = logging.getLogger('glompo.manager')

    def has_cache(self):
        with self.cache_lock:
            return bool(self.cache)

    def put(self, item, block=True, timeout=None):
        """ A put is attempted according to block and timeout. If the cache is being used the item is first added to the
            cache and then the entire cache is put on the queue in accordance with block and timeout.
        """
        if self.fast_func:
            with self.cache_lock:
                self.cache.append(item)
                super().put(self.cache, block, timeout)
                self.cache = []
        else:
            super().put([item], block, timeout)

    def put_nowait(self, item):
        """ A put is attempted but if the queue is Full, instead of raising an exception the item is saved to a cache.
            From then on all items are appended to the cache and put in the queue in chunks of size chunk_size. When the
            cache is full the put will block until the cache has been put on the queue.
        """
        if not self.fast_func:
            try:
                super().put([item], False)
            except queue.Full:
                with self.cache_lock:
                    self.logger.info("Queue caching activated.")
                    self.fast_func = True
                    self.cache.append(item)

        else:
            with self.cache_lock:
                self.cache.append(item)
                if len(self.cache) >= self.chunk_size:
                    super().put(self.cache)
                    self.cache = []

    def put_incache(self, item):
        """ Item is explicitly placed in the cache rather than the queue. If the cache is not yet in use it is
            opened.
        """
        self.logger.info("Queue caching activated.")
        self.fast_func = True
        with self.cache_lock:
            self.cache.append(item)

    def flush(self, block=True, timeout=None):
        """ Attempts to put cache items (if any) in the queue according to block and timeout. """
        if self.fast_func:
            with self.cache_lock:
                super().put(self.cache, block, timeout)
                self.cache = []
