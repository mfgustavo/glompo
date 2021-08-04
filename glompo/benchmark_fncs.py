import warnings
from abc import ABC, abstractmethod
from time import sleep
from typing import Sequence, Tuple, Union

import numpy as np

__all__ = ("BaseTestCase",
           "Ackley",
           "Alpine01",
           "Alpine02",
           "Deceptive",
           "Easom",
           "ExpLeastSquaresCost",
           "Griewank",
           "Langermann",
           "Levy",
           "Michalewicz",
           "Qing",
           "Rana",
           "Rastrigin",
           "Rosenbrock",
           "Schwefel",
           "Shekel",
           "Shubert",
           "Stochastic",
           "StyblinskiTang",
           "Trigonometric",
           "Vincent",
           "ZeroSum",
           )


class BaseTestCase(ABC):
    """ Basic API for Optimization test cases.

    Parameters
    ----------
    dims
        Number of parameters in the input space.
    delay
        Pause (in seconds) between function evaluations to mimic slow functions.
    """

    def __init__(self, dims: int, *, delay: float = 0):
        self._dims = dims
        self._delay = delay

    @property
    def dims(self) -> int:
        """ Number of parameters in the input space. """
        return self._dims

    @property
    @abstractmethod
    def min_x(self) -> Sequence[float]:
        """ The location of the global minimum in parameter space. """

    @property
    @abstractmethod
    def min_fx(self) -> float:
        """ The function value of the global minimum. """

    @property
    @abstractmethod
    def bounds(self) -> Sequence[Tuple[float, float]]:
        """ Sequence of min/max pairs bounding the function in each dimension. """

    @property
    def delay(self) -> float:
        """ Delay (in seconds) between function evaluations to mimic slow functions. """
        return self._delay

    @abstractmethod
    def __call__(self, x: Sequence[float]) -> float:
        """ Evaluates the function.

            Parameters
            ----------
            x
                Vector in parameter space where the function will be evaluated.

            Returns
            -------
            float
                Function value at `x`.

        """
        sleep(self.delay)


class Ackley(BaseTestCase):
    """ Implementation of the Ackley optimization test function [b]_.

    .. math::
       f(x) = - a \\exp\\left(-b \\sqrt{\\frac{1}{d}\\sum^d_{i=1}x_i^2}\\right)
              - \\exp\\left(\\frac{1}{d}\\sum^d_{i=1}\\cos\\left(cx_i\\right)\\right)
              + a
              + \\exp(1)

    Recommended bounds: :math:`x_i \\in [-32.768, 32.768]`

    Global minimum: :math:`f(0, 0, ..., 0) = 0`

    .. image:: /_static/ackley.png
       :align: center
       :alt: Multimodal flat surface with a single deep global minima. Multimodal version of the Easom function.

    Parameters
    ----------
    a
        Ackley function parameter
    b
        Ackley function parameter
    c
        Ackley function parameter
    """

    def __init__(self, dims: int = 2, a: float = 20, b: float = 0.2, c: float = 2 * np.pi, *, delay: float = 0):
        super().__init__(dims, delay=delay)
        self.a, self.b, self.c = a, b, c

    def __call__(self, x) -> float:
        x = np.array(x)
        term1 = -self.a
        sos = 1 / self.dims * np.sum(x ** 2)
        term1 *= np.exp(-self.b * np.sqrt(sos))

        cos = 1 / self.dims * np.sum(np.cos(self.c * x))
        term2 = -np.exp(cos)

        term34 = self.a + np.e

        sleep(self.delay)
        return term1 + term2 + term34

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [(-32.768, 32.768)] * self.dims


class Alpine01(BaseTestCase):
    """ Implementation of the Alpine Type-I optimization test function [a]_.

        .. math::
           f(x) = \\sum^n_{i=1}\\left|x_i\\sin\\left(x_i\\right)+0.1x_i\\right|

        Recommended bounds: :math:`x_i \\in [-10, 10]`

        Global minimum: :math:`f(0, 0, ..., 0) = 0`

        .. image:: /_static/alpine01.png
           :align: center
           :alt: Highly oscillatory non-periodic surface.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        calc = np.sin(x)
        calc *= x
        calc += 0.1 * x
        calc = np.abs(calc)

        return np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [(-10, 10)] * self.dims


class Alpine02(BaseTestCase):
    """ Implementation of the Alpine Type-II optimization test function [a]_.

        .. math::
           f(x) = - \\prod_{i=1}^n \\sqrt{x_i} \\sin{x_i}

        Recommended bounds: :math:`x_i \\in [0, 10]`

        Global minimum: :math:`f(7.917, 7.917, ..., 7.917) = -6.1295`

        .. image:: /_static/alpine02.png
           :align: center
           :alt: Moderately oscillatory periodic surface.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        calc = np.sin(x)
        calc *= np.sqrt(x)

        return -np.prod(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return [7.917] * self.dims

    @property
    def min_fx(self) -> float:
        return -2.808 ** self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 10]] * self.dims


class Deceptive(BaseTestCase):
    """ Implementation of the Deceptive optimization test function [a]_.

    .. math:
       f(x) = - \\left[\\frac{1}{n}\\sum^n_{i=1}g_i\\left(x_i\\right)\\right]

    Recommended bounds: :math:`x_i \\in [0, 1]`

    Global minimum: :math:`f(a) = -1`

    .. image:: /_static/deceptive.png
       :align: center
       :alt: Small global minimum surrounded by areas which slope away from it.

    Parameters
    ----------
    b
        Non-linearity parameter.
    shift_positive
        Shifts the entire function such that the global minimum falls at 0.
    """

    def __init__(self, dims: int = 2, b: float = 2, *, shift_positive: bool = False, delay: float = 0):
        super().__init__(dims, delay=delay)
        self.shift = shift_positive
        self.b = b
        self._min_x = np.random.uniform(0, 1, dims)

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)

        calc = - (1 / self.dims * np.sum(self._g(x))) ** self.b

        if self.shift:
            return calc + 1

        return calc

    def _g(self, vec: np.ndarray):
        """ Sub-calculation of the :meth:`__call__` method which returns :math:`g(x)`. """
        gx = np.zeros(len(vec))

        for i, x in enumerate(vec):
            ai = self._min_x[i]
            if 0 <= x <= 0.8 * ai:
                gx[i] = 0.8 - x / ai
            elif 0.8 * ai < x <= ai:
                gx[i] = 5 * x / ai - 4
            elif ai < x <= (1 + 4 * ai) / 5:
                gx[i] = (5 * (x - ai)) / (ai - 1) + 1
            elif (1 + 4 * ai) / 5 < x <= 1:
                gx[i] = (x - 1) / (1 - ai) + 0.8

        return gx

    @property
    def min_x(self) -> Sequence[float]:
        return self._min_x

    @property
    def min_fx(self) -> float:
        return -1 if not self.shift else 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 1]] * self.dims


class Easom(BaseTestCase):
    """ Implementation of the Easom optimization test function [a]_.

    .. math::
       f(x) = - \\cos\\left(x_1\\right)\\cos\\left(x_2\\right)\\exp\\left(-(x_1-\\pi)^2-(x_2-\\pi)^2\\right)

    Recommended bounds: :math:`x_1,x _2 \\in [-100, 100]`

    Global minimum: :math:`f(\\pi, \\pi) = -1`

    .. image:: /_static/easom.png
       :align: center
       :alt: Totally flat surface with a single very small bullet hole type minimum.

    Parameters
    ----------
    shift_positive
        Shifts the entire function such that the global minimum falls at 0.
    """

    def __init__(self, *args, shift_positive: bool = False, delay: float = 0):
        super().__init__(2, delay=delay)
        self.shift = shift_positive

    def __call__(self, x):
        sleep(self.delay)

        calc = -np.cos(x[0])
        calc *= np.cos(x[1])
        calc *= np.exp(-(x[0] - np.pi) ** 2 - (x[1] - np.pi) ** 2)

        if self.shift:
            return calc + 1

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [np.pi, np.pi]

    @property
    def min_fx(self) -> float:
        return -1 if not self.shift else 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-100, 100]] * 2


class Griewank(BaseTestCase):
    """ Implementation of the Griewank optimization test function [b]_.

        .. math::
           f(x) = \\sum_{i=1}^d \\frac{x_i^2}{4000} - \\prod_{i=1}^d \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1

        Recommended bounds: :math:`x_i \\in [-600, 600]`

        Global minimum: :math:`f(0, 0, ..., 0) = 0`

        .. image:: /_static/griewank.png
           :align: center
           :alt: Highly oscillatory totally-periodic surface on a general parabolic surface. Similar to Rastrigin.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        term1 = 1 / 4000 * np.sum(x ** 2)

        term2 = x / np.sqrt(np.arange(1, len(x) + 1))
        term2 = np.prod(np.cos(term2))

        return 1 + term1 - term2

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-600, 600]] * self.dims


class Langermann(BaseTestCase):
    """ When called returns evaluations of the Langermann function [a]_ [b]_.

    .. math::
      f(x) & = & - \\sum_{i=1}^5 \\frac{c_i\\cos\\left(\\pi\\left[(x_1-a_i)^2 + (x_2-b_i)^2\\right]\\right)}
                                       {\\exp\\left(\\frac{(x_1-a_i)^2 + (x_2-b_i)^2}{\\pi}\\right)}\\\\
      \\mathbf{a} & = & \\{3, 5, 2, 1, 7\\}\\\\
      \\mathbf{b} & = & \\{5, 2, 1, 4, 9\\}\\\\
      \\mathbf{c} & = & \\{1, 2, 5, 2, 3\\}\\\\

    Recommended bounds: :math:`x_1, x_2 \\in [0, 10]`

    Global minimum: :math:`f(2.00299219, 1.006096) = -5.1621259`

    .. image:: /_static/langermann.png
       :align: center
       :alt: Analogous to ripples on a water surface after three drops have hit it.

    Parameters
    ----------
    shift_positive
        Shifts the entire function such that the global minimum falls at ~0.
    """

    def __init__(self, *args, shift_positive: bool = False, delay: float = 0):
        super().__init__(2, delay=delay)
        self.shift = shift_positive

    def __call__(self, x: np.ndarray) -> float:
        a = np.array([3, 5, 2, 1, 7])
        b = np.array([5, 2, 1, 4, 9])
        c = np.array([1, 2, 5, 2, 3])
        x1, x2 = x[0], x[1]

        cos = c * np.cos(np.pi * ((x1 - a) ** 2 + (x2 - b) ** 2))
        exp = np.exp(((x1 - a) ** 2 + (x2 - b) ** 2) / np.pi)

        sleep(self.delay)
        calc = - np.sum(cos / exp)

        if self.shift:
            return calc + 5.2

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [2.00299219, 1.006096]

    @property
    def min_fx(self) -> float:
        return -5.1621259 if not self.shift else 0.0378741

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0, 10]] * 2


class ExpLeastSquaresCost(BaseTestCase):
    """ Least squares type cost function.
    Bespoke test function which takes the form of least squares cost function by solving for the parameters of a
    sum of exponential terms. Compatible with the GFLS solver.

    .. math::
       f(p) & = & \\sum_i^{n} (g - g_{train})^2\\\\
       g(p, u) & = & \\sum_i^d \\exp(-p_i u)\\\\
       g_{train}(p) & = & g(p, u_{train}) \\\\
       u_{train} & = & \\mathcal{U}_{[x_{min}, x_{max}]}

    Recommended bounds: :math:`x_i \\in [-2, 2]`

    Global minimum: :math:`f(p_1, p_2, ..., p_n) \\approx 0`

    .. image:: /_static/lstsqrs.png
       :align: center
       :alt: Minimum sandwiched between very flat surface and very steep walls.

    Parameters
    ----------
    n_train
        Number of training points used in the construction of the error function.
    sigma_eval
        Random perturbations added at the execution of each function evaluation.
        :math:`f = f(1 + \\mathcal{U}_{[-\\sigma_{eval}, \\sigma_{eval}]})`
    sigma_fixed
        Random perturbations added to the construction of the training set so that the
        global minimum error cannot be zero.
        :math:`g_{train} = g_{train}(1 + \\mathcal{U}_{[-\\sigma_{eval}, \\sigma_{eval}]})`
    u_train
        If an int is provided, training points are randomly selected in the interval :math:`[0, u_{train})`.
        If a tuple is provided, training points are randomly selected in the interval :math:`[u_{train,0},
        u_{train,1}]`.
        If an array like object of length >2 is provided then the list is explicitly used as the locations of
        the training points.
    p_range
        Range between which the true parameter values will be drawn.

    Notes
    -----
    The properties :attr:`~BaseTestCase.min_fx` and :attr:`~BaseTestCase.min_x` are only guaranteed for
    `sigma_fixed` = 0; otherwise they are only estimates. This is because the added random noise may create a
    better fit of the data an unknown vector.
    """

    def __init__(self, dims: int = 2, n_train: int = 10, sigma_eval: float = 0,
                 sigma_fixed: float = 0, u_train: Union[int, Tuple[float, float], Sequence[float]] = 10,
                 p_range: Tuple[float, float] = (-2, 2), *, delay: float = 0):
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


class Levy(BaseTestCase):
    """ Implementation of the Levy optimization test function [b]_.

        .. math::
            f(x) & = & \\sin^2(\\pi w_1) + \\sum^{d-1}_{i=1}\\left(w_i-1\\right)^2\\left[1+10\\sin^2\\left(\\pi w_i +1
            \\right)\\right] + \\left(w_d-1\\right)^2\\left[1+\\sin^2\\left(2\\pi w_d\\right)\\right] \\\\
            w_i & = & 1 + \\frac{x_i - 1}{4}

        Recommended bounds: :math:`x_i \\in [-10, 10]`

        Global minimum: :math:`f(1, 1, ..., 1) = 0`

        .. image:: /_static/levy.png
           :align: center
           :alt: Moderately oscillatory periodic surface.
    """

    def __call__(self, x: np.ndarray) -> float:
        x = np.array(x)
        w = self._w(x)

        term1 = np.sin(np.pi * w[0]) ** 2

        term2 = (w - 1) ** 2
        term2 *= 1 + 10 * np.sin(np.pi * w + 1) ** 2
        term2 = np.sum(term2)

        term3 = (w[-1] - 1) ** 2
        term3 *= 1 + np.sin(2 * np.pi * w[-1]) ** 2

        sleep(self.delay)
        return term1 + term2 + term3

    @staticmethod
    def _w(x: np.ndarray) -> np.ndarray:
        return 1 + (x - 1) / 4

    @property
    def min_x(self) -> Sequence[float]:
        return [1] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-10, 10]] * self.dims


class Michalewicz(BaseTestCase):
    """ Implementation of the Michalewicz optimization test function [b]_.

    .. math::
        f(x) = - \\sum^d_{i=1}\\sin(x_i)\\sin^{2m}\\left(\\frac{ix_i^2}{\\pi}\\right)

    Recommended bounds: :math:`x_i \\in [0, \\pi]`

    Global minimum:

    .. math::

        f(x) = \\begin{cases}
                    -1.8013 & \\text{if} & d=2 \\\\
                    -4.687 & \\text{if} & d=5 \\\\
                    -9.66 & \\text{if} & d=10 \\\\
               \\end{cases}

    .. image:: /_static/michalewicz.png
       :align: center
       :alt: Flat surface with many valleys and a single global minimum.

    Parameters
    ----------
    m
        Parametrization of the function. Lower values make the valleys more informative at pointing to the
        minimum. High values (:math:`\\pm10`) create a needle-in-a-haystack function where there is no
        information pointing to the minimum.
    """

    def __init__(self, dims: int = 2, m: float = 10, *, delay: float = 0):
        super().__init__(dims, delay=delay)
        self.m = m

    def __call__(self, x):
        sleep(self.delay)

        i = np.arange(1, len(x) + 1)
        x = np.array(x)

        calc = np.sin(x)
        calc *= np.sin(i * x ** 2 / np.pi) ** (2 * self.m)

        return -np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Global minimum only known for d=2, 5 and 10 but locations are unknown.")
        return [0] * self._dims

    @property
    def min_fx(self) -> float:
        if self._dims == 2:
            return -1.8013
        if self._dims == 5:
            return -4.687658
        if self._dims == 10:
            return -9.66015
        warnings.warn("Global minimum only known for d=2, 5 and 10")
        return np.inf

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [(0, np.pi)] * self.dims


class Qing(BaseTestCase):
    """ Implementation of the Qing optimization test function [a]_.

        .. math::
           f(x) = \\sum^d_{i=1} (x_i^2-i)^2

        Recommended bounds: :math:`x_i \\in [-500, 500]`

        Global minimum: :math:`f(\\sqrt{1}, \\sqrt{2}, ..., \\sqrt{n}) = 0`

        .. image:: /_static/qing.png
           :align: center
           :alt: Globally flat with parabolic walls but has :math:`2^d` degenerate global minima.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)
        i = np.arange(1, self.dims + 1)

        calc = (x ** 2 - i) ** 2

        return np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Global minimum is degenerate at every positive and negative combination of the returned "
                      "parameter vector.", UserWarning)
        return np.sqrt(np.arange(1, self.dims + 1))

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims


class Rana(BaseTestCase):
    """ Implementation of the Rana optimization test function [a]_.

        .. math::
           f(x) = \\sum^d_{i=1}\\left[x_i\\sin\\left(\\sqrt{\\left|x_1-x_i+1\\right|}\\right)
                                      \\cos\\left(\\sqrt{\\left|x_1+x_i+1\\right|}\\right)\\\\
                                 + (x_1+1)\\sin\\left(\\sqrt{\\left|x_1+x_i+1\\right|}\\right)
                                      \\cos\\left(\\sqrt{\\left|x_1-x_i+1\\right|}\\right)
                              \\right]

        Recommended bounds: :math:`x_i \\in [-500.000001, 500.000001]`

        Global minimum: :math:`f(-500, -500, ..., -500) = -928.5478`

        .. image:: /_static/rana.png
           :align: center
           :alt: Highly multimodal and chaotic, optimum is on the lower bound
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        term1 = x
        term1 = term1 * np.sin(np.sqrt(np.abs(x[0] - x + 1)))
        term1 = term1 * np.cos(np.sqrt(np.abs(x[0] + x + 1)))

        term2 = x[0] + 1
        term2 = term2 * np.sin(np.sqrt(np.abs(x[0] + x + 1)))
        term2 = term2 * np.cos(np.sqrt(np.abs(x[0] - x + 1)))

        return np.sum(term1 + term2)

    @property
    def min_x(self) -> Sequence[float]:
        return [-500] * self.dims

    @property
    def min_fx(self) -> float:
        return self.__call__(self.min_x)

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500.000001, 500.000001]] * self.dims


class Rastrigin(BaseTestCase):
    """ Implementation of the Rastrigin optimization test function [b]_.

        .. math::
           f(x) = 10d + \\sum^d_{i=1} \\left[x_i^2-10\\cos(2\\pi x_i)\\right]

        Recommended bounds: :math:`x_i \\in [-5.12, 5.12]`

        Global minimum: :math:`f(0, 0, ..., 0) = 0`

        .. image:: /_static/rastrigin.png
           :align: center
           :alt: Modulation of a unimodal paraboloid with multiple regular local minima.
    """

    def __call__(self, x):
        x = np.array(x)

        calc = 10 * self.dims
        calc += np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

        sleep(self.delay)
        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [0] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-5.12, 5.12]] * self.dims


class Rosenbrock(BaseTestCase):
    """ Implementation of the Rosenbrock optimization test function [b]_.

        .. math::
           f(x) = \\sum^{d-1}_{i=1}\\left[100(x_{i+1}-x_i^2)^2+(x_i-1)^2\\right]

        Recommended bounds: :math:`x_i \\in [-2.048, 2.048]`

        Global minimum: :math:`f(1, 1, ..., 1) = 0`

        .. image:: /_static/rosenbrock.png
           :align: center
           :alt: Global minimum is located in a very easy to find valley but locating it within the valley is difficult.
    """

    def __call__(self, x):
        total = 0
        for i in range(self.dims - 1):
            total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        sleep(self.delay)
        return total

    @property
    def min_x(self) -> Sequence[float]:
        return [1] * self.dims

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-2.048, 2.048]] * self._dims


class Schwefel(BaseTestCase):
    """ Implementation of the Schwefel optimization test function [b]_.

    .. math::
       f(x) = 418.9829d - \\sum^d_{i=1} x_i\\sin\\left(\\sqrt{|x_i|}\\right)

    Recommended bounds: :math:`x_i \\in [-500, 500]`

    Global minimum: :math:`f(420.9687, 420.9687, ..., 420.9687) = -418.9829d`

    .. image:: /_static/schwefel.png
       :align: center
       :alt: Multimodal and deceptive in that the global minimum is very far from the next best local minimum.

    Parameters
    ----------
    shift_positive
        Shifts the entire function such that the global minimum falls at ~0.
    """

    def __init__(self, dims: int = 2, *, shift_positive: bool = False, delay: float = 0):
        super().__init__(dims, delay=delay)
        self.shift = shift_positive

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)
        calc = np.sum(-x * np.sin(np.sqrt(np.abs(x))))

        if self.shift:
            return calc + 418.9830 * self.dims

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [420.9687] * self.dims

    @property
    def min_fx(self) -> float:
        return -418.9829 * self.dims if not self.shift else 0.0001 * self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims


class Shekel(BaseTestCase):
    """ Implementation of the Shekel optimization test function [b]_.

    .. math::
       f(x) = - \\sum^m_{i=1}\\left(\\sum^d_{j=1} (x_j - C_{ji})^2 + \\beta_i\\right)^{-1}

    Recommended bounds: :math:`x_i \\in [-32.768, 32.768]`

    Global minimum: :math:`f(4, 4, 4, 4) =~ -10`

    .. image:: /_static/shekel.png
       :align: center
       :alt: Multiple minima of different depths clustered together on a mostly-flat surface.

    Parameters
    ----------
    m
        Number of minima. Global minimum certified for m=5,7 and 10.
    shift_positive
        Shifts the entire function such that the function is strictly positive.
        Since this is variable for this function the adjustment is +12 and thus the global minimum will not
        necessarily fall at zero.
    """

    def __init__(self, dims: int = 2, m: int = 10, *, shift_positive: bool = False, delay: float = 0):
        assert 0 < dims < 5
        super().__init__(dims, delay=delay)
        self.shift = shift_positive

        if any([m == i for i in (5, 7, 10)]):
            self.m = m

            self.a = np.array([[4] * 4,
                               [1] * 4,
                               [8] * 4,
                               [6] * 4,
                               [3, 7] * 2])
            self.c = 0.1 * np.array([1, 2, 2, 4])

            if m == 5:
                self.c = np.append(self.c, 0.6)
            else:
                self.a = np.append(self.a,
                                   np.array([[2, 9] * 2,
                                             [5, 5, 3, 3]]),
                                   axis=0)
                self.c = np.append(self.c, 0.1 * np.array([4, 6, 3]))
                if m == 10:
                    self.a = np.append(self.a,
                                       np.array([[8, 1] * 2,
                                                 [6, 2] * 2,
                                                 [7, 3.6] * 2]),
                                       axis=0)
                    self.c = np.append(self.c, 0.1 * np.array([7, 5, 5]))

        else:
            raise ValueError("m can only be 5, 7 or 10")

    def __call__(self, x):
        sleep(self.delay)
        x = np.array(x)

        calc = (self.c + np.sum((x - self.a[:, :self.dims]) ** 2, axis=1)) ** -1
        calc = -np.sum(calc)

        if self.shift:
            return calc + 12

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [4] * self.dims

    @property
    def min_fx(self) -> float:
        warnings.warn("Global minimum is only known for some combinations of m and d. The provided value is "
                      "approximate.")
        return -10.6

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-32.768, 32.768]] * self.dims


class Shubert(BaseTestCase):
    """ Implementation of the Shubert Type-I, Type-III and Type-IV optimization test functions [a]_.

    .. math::
       f_I(x) & = & \\sum^2_{i=1}\\sum^5_{j=1} j \\cos\\left[(j+1)x_i+j\\right]\\\\
       f_{III}(x) & = & \\sum^5_{i=1}\\sum^5_{j=1} j \\sin\\left[(j+1)x_i+j\\right]\\\\
       f_{IV}(x) & = & \\sum^5_{i=1}\\sum^5_{j=1} j \\cos\\left[(j+1)x_i+j\\right]\\\\

    Recommended bounds: :math:`x_i \\in [-10, 10]`

    .. image:: /_static/shubert.png
       :align: center
       :alt: Highly oscillatory, periodic surface. Many degenerate global minima regularly placed.

    Parameters
    ----------
    style
        Selection between the Shubert01, Shubert03 & Shubert04 functions. Each more oscillatory than the previous.
    shift_positive
        Shifts the entire function such that the global minimum falls at 0.
    """

    def __init__(self, dims: int = 2, style: int = 1, *, shift_positive: bool = False, delay: float = 0):
        super().__init__(dims, delay=delay)
        self.style = style
        self.shift_positive = shift_positive

    def __call__(self, x) -> float:
        super().__call__(x)

        if self.style == 1:
            x = np.array(x).reshape((-1, 1))
            j = np.arange(1, 6)

            calc = (j + 1) * x + j
            calc = np.cos(calc)
            calc = j * calc
            calc = np.sum(calc, axis=1)
            calc = np.prod(calc)

            if self.shift_positive:
                calc += 186.731

        elif self.style == 3:
            x = np.reshape(x, (-1, 1))
            j = np.arange(1, 6)

            calc = (j + 1) * x + j
            calc = np.sin(calc)
            calc = j * calc
            calc = np.sum(calc)

            if self.shift_positive:
                calc += 24.062499

        else:
            x = np.reshape(x, (-1, 1))
            j = np.arange(1, 6)

            calc = (j + 1) * x + j
            calc = np.cos(calc)
            calc = j * calc
            calc = np.sum(calc)

            if self.shift_positive:
                calc += 29.016015

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        warnings.warn("Degenerate global minima")
        return

    @property
    def min_fx(self) -> float:
        if self._dims > 2:
            warnings.warn("Minimum unknown for d>2")
            return None

        if self.shift_positive:
            return 0

        if self.style == 1:
            return -186.7309
        if self.style == 3:
            return -24.062499
        return -29.016015

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-10, 10]] * self._dims


class Stochastic(BaseTestCase):
    """ Implementation of the Stochastic optimization test function [a]_.

        .. math::
           f(x) & = & \\sum^d_{i=1} \\epsilon_i\\left|x_i-\\frac{1}{i}\\right| \\\\
           \\epsilon_i & = & \\mathcal{U}_{[0, 1]}

        Recommended bounds: :math:`x_i \\in [-5, 5]`

        Global minimum: :math:`f(1/d, 1/d, ..., 1/d) = 0`

        .. image:: /_static/stochastic.png
           :align: center
           :alt: Parabolic function with random evaluation noise making a substantial contribution.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)
        e = np.random.rand(self.dims)
        i = np.arange(1, self.dims + 1)

        calc = e * np.abs(x - 1 / i)

        return np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return 1 / np.arange(1, self.dims + 1)

    @property
    def min_fx(self) -> float:
        return 0

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-5, 5]] * self.dims


class StyblinskiTang(BaseTestCase):
    """ Implementation of the Styblinski-Tang optimization test function [b]_.

        .. math::
           f(x) = \\frac{1}{2}\\sum^d_{i=1}\\left(x_i^4-16x_i^2+5x_i\\right)

        Recommended bounds: :math:`x_i \\in [-500, 500]`

        Global minimum: :math:`f(-2.90, -2.90, ..., -2.90) = -39.16616570377 d`

        .. image:: /_static/styblinskitang.png
           :align: center
           :alt: Similar to Qing function but minima are deceptively similar but not actually degenerate.
    """

    def __call__(self, x) -> float:
        super().__call__(x)
        x = np.array(x)

        calc = x ** 4 - 16 * x ** 2 + 5 * x

        return 0.5 * np.sum(calc)

    @property
    def min_x(self) -> Sequence[float]:
        return [-2.903534018185960] * self.dims

    @property
    def min_fx(self) -> float:
        return -39.16616570377142 * self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims


class Trigonometric(BaseTestCase):
    """ Implementation of the Trigonometric Type-II optimization test function [a]_.

        .. math::
           f(x) = 1 + \\sum_{i=1}^d 8 \\sin^2 \\left[7(x_i-0.9)^2\\right]
           + 6 \\sin^2 \\left[14(x_i-0.9)^2\\right]+(x_i-0.9)^2

        Recommended bounds: :math:`x_i \\in [-500, 500]`

        Global minimum: :math:`f(0.9, 0.9, ..., 0.9) = 1`

        .. image:: /_static/trigonometric.png
           :align: center
           :alt: Parabolic but becomes a multimodal flat surface with many peaks and troughs near the minimum.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        core = (np.array(x) - 0.9) ** 2
        sin1 = 8 * np.sin(7 * core) ** 2
        sin2 = 6 * np.sin(14 * core) ** 2

        total = sin1 + sin2 + core
        return 1 + np.sum(total)

    @property
    def min_x(self) -> Sequence[float]:
        return [0.9] * self.dims

    @property
    def min_fx(self) -> float:
        return 1

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-500, 500]] * self.dims


class Vincent(BaseTestCase):
    """ Implementation of the Vincent optimization test function [a]_.

        .. math::
           f(x) = - \\sum^d_{i=1} \\sin\\left(10\\log(x)\\right)

        Recommended bounds: :math:`x_i \\in [0.25, 10]`

        Global minimum: :math:`f(7.706, 7.706, ..., 7.706) = -d`

        .. image:: /_static/vincent.png
           :align: center
           :alt: 'Flat' surface made of period peaks and trough of various sizes.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        calc = 10 * np.log(x)
        calc = np.sin(calc)
        calc = -np.sum(calc)

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [7.70628098] * self.dims

    @property
    def min_fx(self) -> float:
        return -self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[0.25, 10]] * self.dims


class ZeroSum(BaseTestCase):
    """ Implementation of the ZeroSum optimization test function [a]_.

        .. math::
           f(x) = \\begin{cases}
                        0 & \text{if} \\sum^n_{i=1} x_i = 0 \\\\
                        1 + (10000|\\sum^n_{i=1} x_i = 0|)^{0.5} & \text{otherwise}
                  \\end{cases}

        Recommended bounds: :math:`x_i \\in [-10, 10]`

        Global minimum: :math:`f(x) = 0 \text{where} \\sum^n_{i=1} x_i = 0`

        .. image:: /_static/zerosum.png
           :align: center
           :alt: Single valley of degenerate global minimum results that is not axi-parallel.
    """

    def __call__(self, x) -> float:
        super().__call__(x)

        tot = np.sum(x)

        if tot == 0:
            return 0

        calc = np.abs(tot)
        calc *= 10000
        calc **= 0.5
        calc += 1

        return calc

    @property
    def min_x(self) -> Sequence[float]:
        return [7.70628098] * self.dims

    @property
    def min_fx(self) -> float:
        return -self.dims

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        return [[-10, 10]] * self.dims
