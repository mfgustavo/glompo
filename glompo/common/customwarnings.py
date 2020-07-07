""" Custom warning classes used in several places in the package for better debugging control. """

__all__ = ("NotImplementedWarning",)


class NotImplementedWarning(UserWarning):
    """ Warnings for elements of GloMPO not yet implemented but not critical enough to raise an error. """
