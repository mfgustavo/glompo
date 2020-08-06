""" Decorators and wrappers used throughout GloMPO. """

import inspect
import sys
from functools import wraps


def process_print_redirect(opt_id, func):
    """ Wrapper to redirect a process' output to a designated text file. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        sys.stdout = open(f"glompo_optimizer_printstreams/{opt_id}_printstream.out", "w")
        sys.stderr = open(f"glompo_optimizer_printstreams/{opt_id}_printstream.err", "w")
        func(*args, **kwargs)

    return wrapper


def catch_user_interrupt(func):
    """ Catches a user interrupt signal and exits gracefully. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("Interrupt signal received. Process stopping.")

    return wrapper


def decorate_all_methods(decorator):
    """ Applies a decorator to every method in a class. """

    def apply_decorator(cls):
        for k, f in cls.__dict__.items():
            if inspect.isfunction(f):
                setattr(cls, k, decorator(f))
        return cls

    return apply_decorator
