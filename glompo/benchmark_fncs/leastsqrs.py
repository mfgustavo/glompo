from typing import Sequence, Tuple, Union

import numpy as np

from ._base import BaseTestCase


class ExpLeastSquaresCost(BaseTestCase):
    """ Bespoke test function which takes the form of least squares cost function by solving for the parameters of a
        sum of exponential terms. Compatible with the GFLS solver.

        .. math::
           f(p) & = & \\sum_i^{n} (g - g_{train})^2\\\\
           g(p, u) & = & \\sum_i^d \\exp(-p_i u)\\\\
           g_{train}(p) & = & g(p, u_{train}) \\\\
           u_{train} & = & \\mathcal{U}_{[x_{min}, x_{max}]}

        Recommended bounds: :math:`x_i \\in [-2, 2]`

        Global minimum: :math:`f(p_1, p_2, ..., p_n) \\approx 0`

        .. image:: /_figs/lstsqrs.png
           :align: center
           :alt: Minimum sandwiched between very flat surface and very steep walls.

    """

    def __init__(self, dims: int = 2, n_train: int = 10, sigma_eval: float = 0,
                 sigma_fixed: float = 0, u_train: Union[int, Tuple[float, float], Sequence[float]] = 10,
                 p_range: Tuple[float, float] = (-2, 2), *, delay: float = 0):
        """ Parameters
            ----------
            n_train : int, default=10
                Number of training points used in the construction of the error function.
            sigma_eval : float, default=0
                Random perturbations added at the execution of each function evaluation.
                :math:`f = f(1 + \\mathcal{U}_{[-\\sigma_{eval}, \\sigma_{eval}]})`
            sigma_fixed : float, default=0
                Random perturbations added to the construction of the training set so that the
                global minimum error cannot be zero.
                :math:`g_{train} = g_{train}(1 + \\mathcal{U}_{[-\\sigma_{eval}, \\sigma_{eval}]})`
            u_train : int or list of float, default=10
                If an int is provided, training points are randomly selected in the interval :math:`[0, u_{train})`.
                If a tuple is provided, training points are randomly selected in the interval :math:`[u_{train,0},
                u_{train,1}]`.
                If an array like object of length >2 is provided then the list is explicitly used as the locations of
                the training points.
            p_range : tuple of float, default=[-2, 2]
                Range between which the true parameter values will be drawn.

            Notes
            -----
            The properties min_fx and min_x are only guaranteed for sigma_fixed = 0; otherwise they are only estimates.
            This is because the added random noise may create a better fit of the data an unknown vector.

        """
        super().__init__(dims, delay=delay)
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
        x = np.array(x)
        return np.sum(self.detailed_call(x) ** 2)

    def detailed_call(self, x) -> Sequence[float]:
        """ Returns a sequence of contributions which are squared and summed to get the final function value. """
        super().__call__(x)

        g_x = np.sum(np.exp(np.einsum('j,h->hj', -x, self.u_train)), axis=1)
        error = g_x - self.g_train

        error *= 1 + np.random.uniform(-self.sigma_eval, self.sigma_eval)

        return error

    @property
    def min_x(self) -> Sequence[float]:
        return self.p_true

    @property
    def min_fx(self) -> float:
        return self._min_fx

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [list(self.p_range)] * self.dims
