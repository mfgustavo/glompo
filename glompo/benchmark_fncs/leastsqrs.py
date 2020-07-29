from typing import Sequence, Tuple, Union

import numpy as np

from ._base import BaseTestCase


class ExpLeastSquaresCost(BaseTestCase):
    """ Bespoke test function which takes the form of least squares cost function. Compatible with the GFLS solver. """

    def __init__(self, dims: int = 2, delay: int = 0, n_train: int = 10, sigma_eval: float = 0,
                 sigma_fixed: float = 0, u_train: Union[int, Tuple[float, float], Sequence[float]] = 10,
                 p_range: Tuple[float, float] = (-2, 2)):
        """
        Solves for the parameters of a sum of exponential terms.
        Recommended bounds: [-2, 2] * dims
        Global minimum: f(p1, p2, ..., pn) ~ 0
        Moderately oscillatory periodic surface.

        The objective function:
            f(p) := Sum_i^n_train (g - g_train) ** 2

            g(p, u) := Sum_i^dims exp(-p_i * u)

            g_train(p) := g(p, u_train)
            where u_train are randomly selected and saved evaluation points at initialisation.

        Parameters
        ----------
        dims: int = 2
            Number of dimensions of the problem
        delay: int = 0
            Delay in seconds after the function is called before results are returned.
            Useful to simulate harder problems.
        n_train: int = 10
            Number of training points used in the construction of the error function.
        sigma_eval: float = 0
            Random perturbations added at the execution of each function evaluation.
            f = f * (1 + random(-sigma_eval, sigma_eval))
        sigma_fixed: float = 0
            Random perturbations added to the construction of the training set so that the
            global minimum error cannot be zero.
            f = f * (1 + random(-sigma_eval, sigma_eval))
        u_train: Union[int, Tuple[float, float], Sequence[float]] = 10
            If an int is provided, training points are randomly selected in the interval [0, u_train)
            If a tuple is provided, training points are randomly selected in the interval [u_train[0], u_train[1])
            If an array like object of length >2 is provided then the list is explicitly used as the locations of
            the training points.
        p_range: Tuple[float, float] = [-2, 2]
            Range between which the true parameter values will be drawn.

        Notes
        -----
        The properties min_fx and min_x are only guaranteed for sigma_fixed = 0; otherwise they are only estimates. This
        is because the added random noise may create a better fit of the data an unknown vector.

        """
        self._dims = dims
        self._delay = delay
        self.n_train = n_train
        self.sigma_eval = sigma_eval
        self.sigma_fixed = sigma_fixed
        self.p_range = p_range

        if isinstance(u_train, int):
            self.u_train = np.random.uniform(0, u_train, n_train)
        elif len(u_train) == 2:
            self.u_train = np.random.uniform(u_train[0], u_train[1], n_train)
        else:
            self.u_train = np.array(u_train)

        self.p_true = np.random.uniform(p_range[0], p_range[1], dims)

        g_train_core = np.sum(np.exp(np.einsum('j,h->hj', -self.p_true, self.u_train)), axis=1)
        self.g_train = g_train_core * (1 + np.random.uniform(-self.sigma_fixed, self.sigma_fixed, self.n_train))
        self._min_fx = sum((g_train_core - self.g_train) ** 2)

    def __call__(self, x) -> float:
        return np.sum(self.resids(x) ** 2)

    def resids(self, x) -> Sequence[float]:
        super().__call__(x)

        g_x = np.sum(np.exp(np.einsum('j,h->hj', -x, self.u_train)), axis=1)
        error = g_x - self.g_train

        error *= 1 + np.random.uniform(-self.sigma_eval, self.sigma_eval)

        return error

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def min_x(self) -> Sequence[float]:
        return self.p_true

    @property
    def min_fx(self) -> float:
        return self._min_fx

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [list(self.p_range)] * self.dims

    @property
    def delay(self) -> float:
        return self._delay
