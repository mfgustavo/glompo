""" Decorators and wrappers used throughout GloMPO. """

import inspect
import sys
import importlib
import warnings
from functools import wraps
from pathlib import Path


def process_print_redirect(opt_id, working_dir, func):
    """ Wrapper to redirect a process' output to a designated text file. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        sys.stdout = Path(working_dir, "glompo_optimizer_printstreams", f"printstream_{opt_id:04}.out").open("w")
        sys.stderr = Path(working_dir, "glompo_optimizer_printstreams", f"printstream_{opt_id:04}.err").open("w")
        func(*args, **kwargs)
        sys.stdout.close()
        sys.stderr.close()

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
        for key, func in cls.__dict__.items():
            if inspect.isfunction(func):
                setattr(cls, key, decorator(func))
        return cls

    return apply_decorator


def needs_optional_package(package: str):
    """ Will warn and not allow a function to execute if the package it requires is not available to the system.
        Applied to functions which provide optional GloMPO functionality.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                importlib.import_module(package)
                return func(*args, **kwargs)
            except (ModuleNotFoundError, ImportError):
                warnings.warn(f"Unable to construct checkpoint without {package} package installed.", ResourceWarning)
                return

        return wrapper

    return decorator
