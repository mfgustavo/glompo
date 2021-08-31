from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

__all__ = ("BaseSummarizer",
           "ResidualAnalysis",)


# TODO Identify contributions causing chaos
# TODO Pie chart contributions but at what point? contributions change with x
# TODO Identify invariable contributions
# TODO Class attributes
# TODO Should summarizers be included?

class BaseSummarizer(ABC):
    """  """

    def __init__(self, weights: Optional[np.ndarray] = None, *args, **kwargs):
        self.weights = weights

    @abstractmethod
    def __call__(self, residuals: np.ndarray) -> float:
        """ Converts a vector of residuals to an overall value. """


class SumOfSquares(BaseSummarizer):
    """ Calculates an overall value by taking the sum of the squares of individual residuals. """

    def __call__(self, residuals: np.ndarray) -> float:
        return np.sum(self.weights * residuals ** 2)


class ResidualAnalysis:
    """ Analyzes the behaviour of individual contributions to an overall cost function.
    Designed for use on cost functions which are constructed from several individual residuals (e.g. a sum of square
    errors cost function). This class provides a handful of useful analysis tools to identify residuals which are:

     #. particularly oscillatory,

     #. largely invariable,

     #. negligible contributors to the overall quantity,

     #. disproportionate contributors to the overall quantity.

    Parameters
    ----------
    converter
        Function which accepts a residual vector and returns a single numerical value. This value is typically the one
        which the user eventually intends to minimize.

    Attributes
    ----------
    """

    @property
    def summarizer(self) -> BaseSummarizer:
        """ Function to convert residuals to a single number. Must the API include weights? How are weights
        defined (eg AMS2020.1 and AMS2020.2)
        """

    @property
    def x(self) -> np.ndarray:
        """ :math:`n \\times k` array of input vectors saved in the analysis. """
        return self._x

    @property
    def residuals(self) -> np.ndarray:
        """ :math:`n \\times h` array of function response vectors saved in the analysis. """
        return self._residuals

    @property
    def k(self) -> int:
        """ Pseudonym for :attr:`dims` """
        return self.dims

    @property
    def h(self) -> int:
        """ Pseudonym for :attr:`residual_dims` """
        return self.residual_dims

    @property
    def n(self) -> int:
        """ Returns the number input / output pairs saved in the analysis. """
        return self._x.shape[0]

    def __init__(self, input_dims: int, residual_dims: int):
        self.dims = input_dims
        self.residual_dims = residual_dims

        self._x = np.empty((0, input_dims))
        self._residuals = np.empty((0, residual_dims))

    def add_pt(self, x: np.ndarray, residuals: np.ndarray):
        """ Appends a new point to the analysis.

        Parameters
        ----------
        x
            One or several vectors of function inputs. Shape expected: :math:`n \\times k`.
        residuals
            Vector of function responses on which the analysis will be performed. Shape expected: :math:`n \\times h`.
        """
        x = np.atleast_2d(x)
        residuals = np.atleast_2d(residuals)

        self._x = np.append(self._x, x, axis=0)
        self._residuals = np.append(self._residuals, residuals, axis=0)

    def rebalance(self) -> np.ndarray:
        """ Suggest new weights which will yield a better balance of contributions to the the overall function. """
        # TODO Implement

    def invariable(self) -> np.ndarray:
        """  """
        # TODO Implement
