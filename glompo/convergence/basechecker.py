

""" Abstract checker classes used to construct the convergence criteria. """


from abc import ABC, abstractmethod
import inspect


__all__ = ("BaseChecker",)


class BaseChecker(ABC):
    """ Base checker from which all checkers must inherit to be compatible with GloMPO. """

    def __init__(self):
        self._converged = False

    @abstractmethod
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        """ When called, this method may check any instance variables and any variables within the manager to determine
            if GloMPO has converged and returns a bool.

            Note: For proper functionality, the result of check_convergence must be saved to self._converged before
            returning.
        """

    def __or__(self, other: 'BaseChecker') -> '_AnyChecker':
        return _AnyChecker(self, other)

    def __and__(self, other: 'BaseChecker') -> '_AllChecker':
        return _AllChecker(self, other)

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

    def is_converged_str(self) -> str:
        """ String representation of the object with its convergence result. """
        mess = str(self)
        mess += f" = {self._converged}"
        return mess


class _CombiChecker(BaseChecker):

    def __init__(self, base1: BaseChecker, base2: BaseChecker):
        super().__init__()
        for base in [base1, base2]:
            if not isinstance(base, BaseChecker):
                raise TypeError("_CombiChecker can only be initialised with instances of BaseChecker subclasses.")
        self.base1 = base1
        self.base2 = base2

    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        pass

    def _combi_string_maker(self, keyword: str):
        return f"[{self.base1} {keyword} \n{self.base2}]"

    def _combi_isconverged_maker(self, keyword: str):
        return f"[{self.base1.is_converged_str()} {keyword} \n" \
               f"{self.base2.is_converged_str()}]"


class _AnyChecker(_CombiChecker):
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self._converged = self.base1.check_convergence(manager) or self.base2.check_convergence(manager)
        return self._converged

    def __str__(self):
        return self._combi_string_maker("OR")

    def is_converged_str(self) -> str:
        return self._combi_isconverged_maker("OR")


class _AllChecker(_CombiChecker):
    def check_convergence(self, manager: 'GloMPOManager') -> bool:
        self._converged = self.base1.check_convergence(manager) and self.base2.check_convergence(manager)
        return self._converged

    def __str__(self):
        return self._combi_string_maker("AND")

    def is_converged_str(self) -> str:
        return self._combi_isconverged_maker("AND")
