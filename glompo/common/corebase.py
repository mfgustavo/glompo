

""" Abstract classes used to construct the hunter and convergence bases. """


from abc import ABC, abstractmethod
import inspect


__all__ = ("_CoreBase", "_CombiCore", "_OrCore", "_AndCore")


class _CoreBase(ABC):
    """ Base on which BaseHunter and BaseChecker are built """

    def __init__(self):
        self._last_result = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Main evaluation method to determine the result of the hunt / convergence. """

    def __or__(self, other: '_CoreBase') -> '_OrCore':
        return _OrCore(self, other)

    def __and__(self, other: '_CoreBase') -> '_AndCore':
        return _AndCore(self, other)

    def __str__(self) -> str:
        lst = ""
        signature = inspect.signature(self.__init__)
        for parm in signature.parameters:
            if parm in dir(self):
                lst += f"{parm}={self.__getattribute__(parm)}, "
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
        self._last_result = None


class _CombiCore(_CoreBase):

    def __init__(self, base1: _CoreBase, base2: _CoreBase):
        super().__init__()
        for base in [base1, base2]:
            if not isinstance(base, _CoreBase):
                raise TypeError("_CombiCore can only be initialised with instances of _CoreBase subclasses.")
        self._base1 = base1
        self._base2 = base2

    def __call__(self, *args, **kwargs):
        self.reset()

    def _combi_string_maker(self, keyword: str):
        return f"[{self._base1} {keyword} \n{self._base2}]"

    def _combi_result_string_maker(self, keyword: str):
        return f"[{self._base1.str_with_result()} {keyword} \n" \
               f"{self._base2.str_with_result()}]"

    def reset(self):
        """ Resets _last_result to None. Given that hunter and checkers are evaluated lazily, it is possible for
            misleading results to be returned by str_with_result indicating a hunt has been evaluated when it has not.
            Bases are thus reset before calls to prevent this.
        """
        self._base1._last_result = None
        self._base1.reset()

        self._base2._last_result = None
        self._base2.reset()


class _OrCore(_CombiCore):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self._last_result = self._base1(*args, **kwargs) or self._base2(*args, **kwargs)
        return self._last_result

    def __str__(self):
        return self._combi_string_maker("|")

    def str_with_result(self) -> str:
        return self._combi_result_string_maker("|")


class _AndCore(_CombiCore):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self._last_result = self._base1(*args, **kwargs) and self._base2(*args, **kwargs)
        return self._last_result

    def __str__(self):
        return self._combi_string_maker("&")

    def str_with_result(self) -> str:
        return self._combi_result_string_maker("&")
