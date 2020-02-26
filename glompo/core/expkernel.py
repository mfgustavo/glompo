

from typing import *
from functools import wraps
import numpy as np
import scipy.optimize as sciopt


def cache(func: callable):
    cache_dict = {}

    @wraps(func)
    def wrapper(*args):
        if args in cache_dict:
            return cache_dict[args]
        else:
            result = func(*args)
            cache_dict[args] = result
            return result

    return wrapper


class ExpKernel:
    """ Implements and calculates instances of the exponential kernel described in Swerszky et al. (2014). """

    def __init__(self, alpha: float, beta: float, rho: float = 1):
        """ Initialises the kernel hyper-parameters.

            Parameters:
            -----------
            alpha : float
                Shape parameter.
            beta : float
                Rate parameter.
        """
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x1: float, x2: float) -> float:
        calc = self.beta ** self.alpha
        calc /= (x1 + x2 + self.beta) ** self.alpha
        return calc

    # TODO Check the eigenvalues of the final solution. If they are not all the same sign then it can indicate a
    #  saddle point and problem with tyhe solution. Consider then forcing a restart but bounding away from the
    #  incumbent solution.
    def optimize_hyperparameters(self,
                                 time_series: np.ndarray,
                                 loss_series: np.ndarray,
                                 noise: Optional[bool] = True,
                                 bounds: Optional[Sequence[Tuple[float, float]]] = None,
                                 x0: Optional[Sequence[float]] = None,
                                 verbose: Optional[bool] = True) -> Union[Tuple[float, float, float], None]:
        """ Maximises the log-marginal likelihood of the kernel with respect to the hyperparameters alpha, beta and
        sigma. Alpha and beta are updated in place and alpha, beta and sigma are all returned.

        Parameters
        ----------
        time_series : np.ndarray
            Independent training variables used for the optimisation.
        loss_series : np.ndarray
            Dependent training variables used for the optimisation.
        noise : Optional[bool]
            If True the method also optimises for sigma, the standard deviation of the measurement error.
        bounds : Optional[Sequence[Tuple[float, float]]]
            Bounds on the optimized hyperparameters of shape (n, 2) where n is two or three depending on the number of
            hyperparameters being optimzed i.e. with or without noise. Each individual tuple takes the form
            (xmin, xmax).
        x0 : Optional[Sequence[float]]
            Tuple of shape n representing the first starting point of the optimization. Restart values are chosen
            randomly based on bounds.
        verbose : Optional[bool]
            If True prints status messages during the optimization, prints nothing otherwise

        Returns
        -------
        alpha : float
            Shape parameter of the covariance function.
        beta : float
            Rate parameter of the covariance function.
        sigma : float
            Noise parameter of the covariance function.
        """

        # Functions used to calculate the optimal hyperparameters. Taken from GPRML.
        def kernel(a: float, b: float, s: float, t1: float, t2: float, i: int, j: int) -> float:
            calc = (b / (t1 + t2 + b)) ** a
            calc += s ** 2 * int(i == j)
            return calc

        def dk_da(a: float, b: float, t1: float, t2: float) -> float:
            frac = b / (t1 + t2 + b)
            calc = frac ** a * np.log(frac)
            return calc

        def dk_db(a: float, b: float, t1: float, t2: float) -> float:
            denom = t1 + t2 + b
            frac1 = b / denom
            frac2 = 1 / denom
            frac3 = b / (denom ** 2)
            calc = a * frac1 ** (a - 1)
            calc *= frac2 - frac3
            return calc

        def dk_ds(s: float, i: int, j: int) -> float:
            return 2 * s * int(i == j)

        @cache
        def construct_matrix(method, *args) -> np.ndarray:
            mat = np.zeros((len_t, len_t), dtype=float)
            for i in range(len_t):
                for j in range(i, len_t):
                    if method is dk_da or method is dk_db:
                        mat[i, j] = method(*args, t[i], t[j])
                    elif method is dk_ds:
                        mat[i, j] = method(*args, i, j)
                    else:
                        mat[i, j] = method(*args, t[i], t[j], i, j)
                    mat[j, i] = mat[i, j]
            return mat

        def kernel_matrix(a: float, b: float, s: float) -> np.ndarray:
            return construct_matrix(kernel, a, b, s)

        def dk_da_matrix(a: float, b: float) -> np.ndarray:
            return construct_matrix(dk_da, a, b)

        def dk_db_matrix(a: float, b: float) -> np.ndarray:
            return construct_matrix(dk_db, a, b)

        def dk_ds_matrix(s: float) -> np.ndarray:
            return construct_matrix(dk_ds, s)

        @cache
        def inv_matrix(a: float, b: float, s: float) -> np.ndarray:
            mat = kernel_matrix(a, b, s)
            mat = np.linalg.inv(mat)
            return mat

        @cache
        def zeta_matrix(a: float, b: float, s: float) -> np.ndarray:
            return np.matmul(inv_matrix(a, b, s), y)

        # Gradient of the log marginal likelihood used as the Jacobian in the maximisation
        @cache
        def d_log_marg_likelihood(a: float, b: float, s: float) -> float:
            Z = zeta_matrix(a, b, s)
            Zt = np.transpose(Z)
            calc = np.matmul(Z, Zt)
            calc -= inv_matrix(a, b, s)

            result = np.zeros(3)
            for i, deriv in enumerate((dk_da_matrix, dk_db_matrix)):
                result[i] = np.trace(np.matmul(calc, deriv(a, b)))

            if noise:
                result[2] = np.trace(np.matmul(calc, dk_ds_matrix(s)))
                return result * 0.5
            else:
                return result[:2] * 0.5

        # Log marginal likelihood (function to be maximised)
        @cache
        def log_marg_likelihood(a: float, b: float, s: float) -> float:
            terms = np.zeros(3)

            calc = inv_matrix(a, b, s)
            calc = np.matmul(calc, y)
            calc = np.matmul(np.transpose(y), calc)
            terms[0] = -0.5 * calc

            _, terms[1] = np.linalg.slogdet(kernel_matrix(a, b, s))
            terms[1] *= -0.5

            terms[2] = -len_t / 2 * np.log(2 * np.pi)

            return np.sum(terms)

        def gen_start() -> tuple:
            a0 = np.clip(np.random.lognormal(0, 1.5), bnds[0][0], bnds[0][1])
            b0 = np.clip(np.random.lognormal(0, 0.25), bnds[1][0], None)
            calc = a0, b0
            if noise:
                sig0 = np.random.uniform(bnds[2][0], bnds[2][1])
                calc = (*calc, sig0)
            return calc

        assert len(time_series) == len(loss_series)

        len_t = len(time_series)
        t = np.reshape(time_series, (-1, 1))
        y = np.reshape(loss_series, (-1, 1))

        if bounds is None:
            bnds = ((0.1, 2), (0.1, 5))
            if noise:
                bnds = (*bnds, (0.0001, 0.01))
        else:
            bnds = bounds

        for attempt in range(10):
            if attempt == 0 and x0 is not None:
                start = x0
            else:
                start = gen_start()
            print(f"Initial Point: {start}") if verbose else None
            res = sciopt.minimize(fun=lambda a: -log_marg_likelihood(a[0], a[1], a[2] if noise else 0),  # Note negative
                                  x0=start,
                                  jac=lambda a: -d_log_marg_likelihood(a[0], a[1], a[2] if noise else 0),
                                  method='trust-constr',
                                  bounds=bnds,
                                  options={'gtol': 1e-12,
                                           'xtol': 1e-6})

            if res.success and verbose:
                print("------------------------------------------------")
                print("Hyperparameters Successfully Optimized")
                print("------------------------------------------------")
                print(f"\u03B1                = {res.x[0]:.4f} Bounds: {bnds[0]}")
                print(f"\u03B2                = {res.x[1]:.4f} Bounds: {bnds[1]}")
                print(f"\u03B3                = {res.x[2]:.4f} Bounds: {bnds[2]}") if noise else None
                print("")
                print(f"fmax             = {-res.fun}")
                print(f"Opti. Iterations = {res.nit}")
                print(f"Func. Evals      = {res.nfev}")
                print(f"Jaco. Evals      = {res.njev}")
                print(f"flag             = {res.message}")
                print("------------------------------------------------")
                self.alpha, self.beta = res.x[:2]
                return res.x if noise else (res.x[0], res.x[1], 0)
            else:
                if attempt < 9 and verbose:
                    print(f"Optimization Failed. Starting {attempt + 1} of 10.")
                else:
                    print(f"Optimization Failed. Aborting. Parameters unchanged.") if verbose else None
                    return None
