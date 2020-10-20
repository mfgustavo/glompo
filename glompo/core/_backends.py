""" Contains code to support using both multiprocessing and threading within the GloMPO manager. """

import sys
import threading
import warnings
from os.path import join as pjoin
from typing import TextIO


class CustomThread(threading.Thread):
    """ Adds an exitcode property to the Thread base class as well as code to redirect the thread printstream to a
        file if this has been setup before hand.
    """

    def __init__(self, *args, redirect_print: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exitcode = 0
        self.redirect = redirect_print

    def run(self):
        try:
            if self.redirect:
                try:
                    opt_id = int(self.name.replace("Opt", ""))
                    sys.stdout.register(opt_id, 'out')
                    sys.stderr.register(opt_id, 'err')
                except AttributeError:
                    warnings.warn("Prinstream redirect failed. Print stream will only redirect if ThreadPrintRedirect "
                                  "is setup beforehand.", RuntimeWarning)
            super().run()
            if self.redirect:
                try:
                    sys.stdout.close(threading.currentThread().ident)
                    sys.stderr.close(threading.currentThread().ident)
                except Exception as e:
                    warnings.warn(f"Closing printstream files failed. Caught exception: {e}", RuntimeWarning)
        except Exception as e:
            self.exitcode = -1
            raise e


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

    def register(self, opt_id: int, ext: str):
        """ Adds a thread to the set of files. """
        thread_id = threading.currentThread().ident
        self.threads[thread_id] = open(pjoin("glompo_optimizer_printstreams", f"printstream_{opt_id:04}.{ext}"), "w+")

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
