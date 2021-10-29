""" Defines the minimization task API. """

__all__ = ('BaseFunction',)

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import tables as tb


class BaseFunction:
    """ Direct use of this class for the minimization task is *not required*. However, this class does define the
        minimum and optional API, and can be helpful in creating a cost function.
    """

    @abstractmethod
    def __call__(self, x: Sequence[float]) -> float:
        """ Minimum cost function requirement.
        Accepts a parameter vector x and returns the function evaluation's result.

        Parameters
        ----------
        x
            Vector in parameter space at which to evaluate the function.

        Returns
        -------
        float
            Result of the function evaluation which is trying to be minimized.
        """

    def detailed_call(self, x: Sequence[float]) -> Sequence[Any]:
        """ Optional function evaluation method.
        When called with a parameter vector (`x`) it returns a sequence of data. The first element of this sequence is
        expected to be the function evaluation result (as returned by :meth:`__call__`). Subsequent elements of the
        sequence may take any form. This function may be used to return information needed by the optimizer algorithm or
        extra information which will be added to the log.

        Parameters
        ----------
        x
            Vector in parameter space at which to evaluate the function.

        Returns
        -------
        float
            Result of the function evaluation which is trying to be minimized.
        *args
            Additional returns of any type and length.
        """
        raise NotImplementedError

    def headers(self) -> Dict[str, tb.Col]:
        """ Optional implementation.
        If :meth:`detailed_call` is being used, this method returns a dictionary descriptor for each column of the
        return. Keys represent the name of each element of the return, values represent the corresponding
        :class:`tables.Col` data type.

        If headers is not defined, GloMPO will attempt to infer types from a function evaluation return. Be warned that
        this is a risky approach as incorrect inferences could be made. Numerical data types are also set to the largest
        possible type (i.e. :code:`float64`) and strings are limited to 280 characters. This may lead to inefficient use
        of space or data being truncated. If :meth:`detailed_call` is being used, implementation of headers is strongly
        recommended.

        Returns
        -------
        Dict[str, :class:`tables.Col`]
            Mapping of heading names to the :class:`tables.Col` type which indicates the type of
            data the column of information will store.

        Examples
        --------
        >>> import tables
        >>> header = {'fx': tables.Float64Col(),
        ...           'training_set_residuals': tables.Float64Col(shape=100),
        ...           'validation_set_fx': tables.Float64Col(),
        ...           'errors': tables.StringCol(itemsize=280, dflt=b'None')}
        """
        raise NotImplementedError

    def checkpoint_save(self, path: Union[str, Path]):
        """ Persists the function into a file or files from which it can be reconstructed.

        This method is used when a checkpoint of the manager is made and the function cannot be persisted directly.
        A checkpoint is a compressed directory of files which persists all aspects of an in-progress optimization.
        These checkpoints can be loaded by :class:`.GloMPOManager` and the optimization resumed.

        Implementing this function is optional and only required if directly pickling the function is not possible.
        In order to load a checkpoint in which :meth:`checkpoint_save` was used, see
        :meth:`.GloMPOManager.load_checkpoint`).

        Parameters
        ----------
        path
            :obj:`str` or :class:`python:pathlib.Path` to a directory into which files will be saved.
        """
        raise NotImplementedError

    @classmethod
    def checkpoint_load(cls, path: Union[str, Path]):
        """ Creates an instance of the :class:`BaseFunction` from sources.

        These source are the products of :meth:`checkpoint_save`. In order to use this method, it should be sent to the
        :code:`task_loader` argument of :meth:`.GloMPOManager.load_checkpoint`.

        Parameters
        ----------
        path
            :obj:`str` or :class:`~python:pathlib.Path` to a directory which contains the files which will be loaded.
        """
        raise NotImplementedError
