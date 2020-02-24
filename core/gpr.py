

import numpy as np
from typing import *

class GaussianProcessRegression:
    """ A class which creates and calculates a Gaussian Process Regression """

    def __init__(self,
                 kernel: Callable[[float, float], float],
                 dims: int,
                 sigma_noise: Optional[float] = 0,
                 mean: Optional[float] = 0,
                 cache_results: Optional[bool] = True):
        """ Sets up the Gaussian Process Regression with its kernel.

            Parameters:
            -----------
            kernel : Callable[[float, float], float]
                A function which return the covariance between two points.
            dims : int
                The number of dimensions in the input space.
            sigma_noise : Optional[float]
                The standard deviation of noise in the known points.
            mean : Optional[float]
                The value to which the process defaults. Can accept a float value (defaulting to zero) or None.
                If None the mean function is assumed to be an unknown and independent of x. It is calculated as a
                free parameter in the regression with a totally uninformed prior.
            cache_results : Optional[bool]
                If True the results of some matrix constructions and inversions are cached. This can increase speed
                but requires more memory.
        """
        self.kernel = kernel
        self.dims = dims
        self.training_pts = {}
        self.sigma_noise = sigma_noise
        self.mean = mean
        # mean_estimate and mean_uncertainty are approximated by the regression
        self.mean_estimate = None
        self.mean_uncertainty = None
        # Caches for repeatedly constructed and inverted matrices
        self._cache_results = cache_results
        self._kernel_cache = {}
        self._inv_kernel_cache = {}

    def _calc_kernels(self, x: np.ndarray) -> np.ndarray:
        """ Returns the mean and covariance matrices using the test points in x. """
        test_count = len(x)
        train_count = len(self.training_pts)

        train_x = np.array([*self.training_pts])
        train_y = np.array([*self.training_pts.values()])

        k_test_train = self._kernel_matrix(x, train_x)
        k_train_test = np.transpose(k_test_train)
        k_test_test = self._kernel_matrix(x, x)
        invK = self._inv_kernel_matrix(train_x, train_x, self.sigma_noise)

        calc_core = np.matmul(k_test_train, invK)

        # Calculation of the covariance function
        covar_fun = k_test_test - np.matmul(calc_core, k_train_test)  # + 1e-6 * np.identity(test_count)

        # Calculation of the mean function
        if self.mean is not None:
            mean_vec_test = np.full((test_count, 1), self.mean)
            mean_vec_train = np.full((train_count, 1), self.mean)
            mean_func = mean_vec_test + np.matmul(calc_core, train_y - mean_vec_train)
        else:
            mean_func = np.matmul(calc_core, train_y)

            ones_test_len = np.ones((1, test_count))
            ones_train_len = np.ones((1, train_count))
            # Calculate R matrix
            r_mat = np.matmul(invK, k_train_test)
            r_mat = np.matmul(ones_train_len, r_mat)
            r_mat = ones_test_len - r_mat

            self.estimate_mean()
            mean_func += self.mean_estimate * np.transpose(r_mat)
            covar_correction = self.mean_uncertainty * np.matmul(np.transpose(r_mat), r_mat)
            covar_fun += covar_correction

        return mean_func, covar_fun

    def add_known(self, x: np.ndarray, f: np.ndarray):
        """ Adds points known from the real function to the regression. """
        x_nest = np.reshape(x, (-1, self.dims))
        f_nest = np.reshape(f, (-1, 1))
        for i, pt in enumerate(x_nest):
            self.training_pts[tuple(pt)] = f_nest[i]

    def sample(self, x: np.ndarray) -> np.ndarray:
        """ Return a sample of the Gaussian Process at the point/s in x. """
        # TODO: Rewrite to deal with tuple inputs
        x_nest = np.reshape(x, (-1, self.dims))
        test_count = len(x_nest)

        mean_func, covar_fun = self._calc_kernels(x_nest)
        chol_mat = np.linalg.cholesky(covar_fun)
        rand = np.random.normal(0, 1, test_count)

        return mean_func.flatten() + np.matmul(chol_mat, rand)

    def sample_all(self, x: np.ndarray) -> np.ndarray:
        """ Returns the current mean function and a single standard deviation confidence interval around it at the
            points in x. Returns a tuple in the order (mean, sd)
        """
        # TODO: Rewrite to deal with tuple inputs
        # Nest x if it is one point or one dimension
        x_nest = np.reshape(x, (-1, self.dims))
        mean_func, covar_fun = self._calc_kernels(x_nest)
        return mean_func.flatten(), np.sqrt(np.diag(covar_fun))

    def estimate_mean(self) -> Tuple[float, float]:
        ones = np.ones(len(self.training_pts))
        train_x = tuple([*self.training_pts])  # Tuple used as dict key in the cache decorator
        train_y = np.reshape([*self.training_pts.values()], (-1, 1))

        invK = self._inv_kernel_matrix(train_x, train_x, self.sigma_noise)

        calc = np.matmul(ones, invK)
        calc = np.matmul(calc, np.transpose(ones))
        calc = calc ** -1
        self.mean_uncertainty = calc

        calc = calc * ones
        calc = np.matmul(calc, invK)
        calc = np.matmul(calc, train_y)
        self.mean_estimate = calc[0]

        return self.mean_estimate, self.mean_uncertainty

    def _kernel_matrix(self, x1, x2) -> np.ndarray:
        vec1 = np.reshape(x1, (-1, self.dims))
        vec2 = np.reshape(x2, (-1, self.dims))
        mat = np.zeros((len(vec1), len(vec2)))

        symmetric = np.array_equal(vec1, vec2)

        if symmetric:
            key = tuple(map(tuple, vec1))
            if key in self._kernel_cache and self._cache_results:
                return self._kernel_cache[key]
            else:
                for i in range(len(vec1)):
                    for j in range(i, len(vec1)):
                        mat[i, j] = self.kernel(vec1[i], vec2[j])
                        mat[j, i] = mat[i, j]
                self._kernel_cache[key] = mat
        else:
            for i in range(len(vec1)):
                for j in range(len(vec2)):
                    mat[i, j] = self.kernel(vec1[i], vec2[j])
        return mat

    def _inv_kernel_matrix(self, x1, x2, sigma) -> np.ndarray:
        key = tuple(map(tuple, x1)), sigma
        if key in self._inv_kernel_cache and self._cache_results:
            return self._inv_kernel_cache[key]
        else:
            mat = self._kernel_matrix(x1, x2) + sigma ** 2 * np.identity(len(x1))
            mat = np.linalg.inv(mat)
            self._inv_kernel_cache[key] = mat
            return mat

    # TODO To implement below look at Hessian update rules for matrix inversion in QM codes BUT this is not a
    #  priority.
    # # BELOW FUNCTION DOES NOT WORK AT EVEN MODERATE DIMENSIONS :(
    # def _inv_kernel_training_matrix(self):
    #     """ Calculates the inverse of the kernel matrix of training points. This is done blockwise according to the
    #     Woodbury matrix identity since the update of the inverse is significantly faster than inverting the entire
    #     matrix from scratch again as it uses the previously calculated inverse of training points. The final matrix
    #     far is cached."""
    #
    #     n_train_pts = len(self.training_pts)
    #     x_train_pts = np.array([*self.training_pts])
    #
    #     if len(self._matrix_cache[0]) == 0:
    #         self._matrix_cache = 1 / (self._kernel_matrix(x_train_pts[0], x_train_pts[0]) + self.sigma_noise ** 2)
    #         n_matrix_cache = 1
    #     else:
    #         n_matrix_cache = len(self._matrix_cache)
    #
    #     for i in range(n_matrix_cache, n_train_pts):
    #         invA = self._matrix_cache
    #         b = np.array([self.kernel(x_train_pts[i], j) for j in x_train_pts[:i+1]])
    #         d = b[-1][0] + self.sigma_noise ** 2
    #         invd = d ** -1
    #         b = b[:-1]
    #         c = np.transpose(b)
    #
    #         print(np.linalg.det(np.linalg.inv(invA) - invd * np.matmul(b, c)))
    #
    #         # Core matrix calculation
    #         scalar = np.matmul(c, invA)
    #         scalar = np.matmul(scalar, b)
    #         scalar = (d - scalar) ** -1
    #         calc = np.matmul(b, c)
    #         calc = np.matmul(invA, calc)
    #         calc = np.matmul(calc, invA)
    #         E = invA + scalar * calc
    #
    #         # New inverse
    #         upper_left = E
    #         upper_right = - invd * np.matmul(E, b)
    #         lower_left = np.transpose(upper_right)
    #         calc = - invd * np.matmul(c, upper_right)
    #         lower_right = invd + calc
    #
    #         self._matrix_cache = np.block([[upper_left, upper_right],
    #                                        [lower_left, lower_right]])
    #
    #         # Test method
    #         kmat = np.zeros((n_train_pts, n_train_pts))
    #         for h, hx in enumerate(x_train_pts):
    #             for k, kx in enumerate(x_train_pts):
    #                 kmat[h, k] = self.kernel(hx, kx)
    #         kmat = np.linalg.inv(kmat)
    #         if np.allclose(self._matrix_cache, kmat):
    #             print("Passed")
    #         else:
    #             print(f"Failed ({np.max(self._matrix_cache - kmat)})")
    #
    #     return self._matrix_cache
