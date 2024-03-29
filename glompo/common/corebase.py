""" Abstract classes used to construct the :class:`BaseHunter` and :class:`BaseChecker` classes. """

import inspect
from abc import ABC, abstractmethod

__all__ = ("_CoreBase", "_CombiCore", "_OrCore", "_AndCore")

from typing import Generator, Iterable


class _CoreBase(ABC):
    """ Base on which :class:`.BaseHunter` and :class:`.BaseChecker` are built. """

    @property
    def last_result(self):
        """ The result of the last :meth:`__call__`. """
        return self._last_result

    @last_result.setter
    def last_result(self, val: bool):
        self._last_result = val

    def __init__(self):
        self._last_result = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Main evaluation method to determine the result of the hunt / convergence. """

    def __iter__(self) -> Iterable['_CoreBase']:
        """ Provides a (flattened) iteration through all the bases which comprise the combined hunter/checker. """
        return iter([self])

    def __str__(self) -> str:
        """ Produces a string of the hunter/checker's name and configuration. """
        lst = ""
        signature = inspect.signature(self.__init__)
        for parm in signature.parameters:
            if parm in dir(self):
                lst += f"{parm}={getattr(self, parm)}, "
            else:
                lst += f"{parm}, "
        lst = lst[:-2]
        return f"{self.__class__.__name__}({lst})"

    def str_with_result(self) -> str:
        """ String representation of the object with its convergence result. """
        mess = str(self)
        mess += f" = {self._last_result}"
        return mess

    def reset(self):
        """ Clears previous evaluation result to avoid misleading printing.
        Resets :attr:`last_result` to :obj:`None`. Given that hunters and checkers are evaluated lazily, it is possible
        for misleading results to be returned by :meth:`str_with_result` indicating a hunt/check has been evaluated when
        it has not. Bases are thus reset before :meth:`__call__` to prevent this.
        """
        self._last_result = None


class _CombiCore(_CoreBase):
    """ Class to handle the AND/OR combination of two :class:`_CoreBase`\\s. """

    def __init__(self, base1: _CoreBase, base2: _CoreBase):
        super().__init__()
        for base in [base1, base2]:
            if not isinstance(base, _CoreBase):
                raise TypeError("_CombiCore can only be initialised with instances of _CoreBase subclasses.")
        self._base1 = base1
        self._base2 = base2
        self._index = -1

    def __call__(self, *args, **kwargs):
        self.reset()

    def _combi_string_maker(self, keyword: str):
        return f"[{self._base1} {keyword} \n{self._base2}]"

    def _combi_result_string_maker(self, keyword: str):
        return f"[{self._base1.str_with_result()} {keyword} \n" \
               f"{self._base2.str_with_result()}]"

    def reset(self):
        self._base1._last_result = None
        self._base1.reset()

        self._base2._last_result = None
        self._base2.reset()

    def __iter__(self) -> Generator[_CoreBase, None, None]:
        return self._bases()

    def _bases(self):
        """ Returns a generator which yields each of the bases which make up the _CombiCore. This is fully recursive
            but the returns are 'flat' (i.e. nesting is not preserved).
        """
        for base in (self._base1, self._base2):
            if isinstance(base, _CombiCore):
                for item in list(base._bases()):
                    yield item
            else:
                yield base


class _OrCore(_CombiCore):
    """ :class:`_CombiCore` which specifically handles OR combinations of :class:`._CoreBase`\\s. """

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self._last_result = self._base1(*args, **kwargs) or self._base2(*args, **kwargs)
        return self._last_result

    def __str__(self):
        return self._combi_string_maker("|")

    def str_with_result(self) -> str:
        return self._combi_result_string_maker("|")


class _AndCore(_CombiCore):
    """ :class:`_CombiCore` which specifically handles AND combinations of :class:`._CoreBase`\\s. """

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self._last_result = self._base1(*args, **kwargs) and self._base2(*args, **kwargs)
        return self._last_result

    def __str__(self):
        return self._combi_string_maker("&")

    def str_with_result(self) -> str:
        return self._combi_result_string_maker("&")
