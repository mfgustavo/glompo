""" Defines the minimization task API. """

__all__ = ('BaseFunction',)

from abc import abstractmethod
from typing import Any, Dict, Sequence

import tables as tb


class BaseFunction:
    """ Direct use of this class for the minimization task is **not required**. However, this class does define the
        minimum and optional API, and can be helpful in creating a cost function.
    """

    @abstractmethod
    def __call__(self, x: Sequence[float]) -> float:
        """ Minimum cost function requirement. Accepts a parameter vector x and returns the function evaluation's
            result.

            Parameters
            ----------
            x : list of float
                Vector in parameter space at which to evaluate the function.

            Returns
            -------
            float
                Result of the function evaluation which is trying to be minimized.
        """

    def detailed_call(self, x: Sequence[float]) -> Sequence[Any]:
        """ Optional function evaluation method. When called with a parameter vector (`x`) it returns a sequence of
            data. The first element of this sequence is expected to be the function evaluation result (as returned by
            the `__call__` method). Subsequent elements of the sequence may take any form. This function may be used to
            return information needed by the optimizer algorithm or extra information which will be added to the log.

            Parameters
            ----------
            x : list of float
                Vector in parameter space at which to evaluate the function.

            Returns
            -------
            float
                Result of the function evaluation which is trying to be minimized.
            ...
                Additional returns of any type and length.
        """
        raise NotImplementedError

    def headers(self) -> Dict[str, tb.Col]:
        """ Optional implementation. If detailed_call is being used, this method returns a dictionary descriptor for
            each column of the return. Keys represent the name of each element of the return, values represent the
            corresponding `tables.Col` data type. See `PyTables` documentation on how to define this dictionary.

            If headers is not defined, GloMPO will attempt to infer types from a function evaluation return. Be warned
            that this is a risky approach as incorrect inferences could be made. Numerical data types are also set to
            the largest possible type (i.e. `float64`) and strings are limited to 280 characters. This may lead to
            inefficient use of space or data being truncated. If `detailed_call` is being used, implementation of
            headers is strongly recommended.

            Returns
            -------
            dict of str to `tables.Col`
                Mapping of heading names to the `tables.Col` type which indicates the type of data the column of
                information will store.

            Examples
            --------
            .. code-block:: python

               header = {'fx': tables.Float64Col(),
                         'training_set_residuals': tables.Float64Col(shape=100),
                         'validation_set_fx': tables.Float64Col(),
                         'errors': tables.StringCol(itemsize=280, dflt=b'None')}
        """
        raise NotImplementedError
