

import numpy as np
from typing import *


class GaussianProcessRegression:
    """ A class which creates and calculates a Gaussian Process Regression """

    def __init__(self,
                 kernel: Callable[[float, float], float],
                 dims: int,
                 sigma_noise: Optional[float] = 0,
                 mean: Optional[float] = None,
                 cache_results: Optional[bool] = True,
                 normalisation_constants: Optional[Tuple[float, float]] = (0, 1)):
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
            normalisation_constants : Optional[Tuple[float, float]]
                Data fed into the GPR is automatically normalised using the tuple provided in the form of (mean, stdev).
                Data will remain unchanged if the default (0, 1) is used.
        """
        self.kernel = kernel
        self.dims = dims
        self._training_pts = {}
        self.sigma_noise = sigma_noise
        self.normalisation_constants = normalisation_constants
        self.mean = self._normalise(mean) if mean else mean
        # Caches for repeatedly constructed and inverted matrices
        self._cache_results = cache_results
        self._kernel_cache = {}
        self._inv_kernel_cache = {}

    def _normalise(self, y: np.ndarray) -> np.ndarray:
        return (y - self.normalisation_constants[0]) / self.normalisation_constants[1]

    def _denormalise(self, y: np.ndarray) -> np.ndarray:
        return y * self.normalisation_constants[1] + self.normalisation_constants[0]

    def _calc_kernels(self, x: np.ndarray) -> np.ndarray:
        """ Returns the mean and covariance matrices using the test points in x. """
        test_count = len(x)
        train_count = len(self._training_pts)

        train_x = np.array([*self._training_pts])
        train_y = np.array([*self._training_pts.values()])

        k_test_train = self._kernel_matrix(x, train_x)
        k_train_test = np.transpose(k_test_train)
        k_test_test = self._kernel_matrix(x, x)

        invK = self._inv_kernel_matrix(train_x, train_x,
                                       self.sigma_noise + 1e-4)  # NB Added here for numerical stability

        calc_core = np.matmul(k_test_train, invK)

        # Calculation of the covariance function
        covar_fun = k_test_test - np.matmul(calc_core, k_train_test)

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

            mean_estimate, mean_uncertainty = self.estimate_mean(True)
            mean_func += mean_estimate * np.transpose(r_mat)
            covar_correction = mean_uncertainty * np.matmul(np.transpose(r_mat), r_mat)
            covar_fun += covar_correction

        return mean_func, covar_fun

    def add_known(self, x: np.ndarray, f: np.ndarray):
        """ Adds points known from the real function to the regression. """
        x_nest = np.reshape(x, (-1, self.dims))
        f_nest = np.reshape(f, (-1, 1))
        f_nest = self._normalise(f_nest)

        for i, pt in enumerate(x_nest):
            self._training_pts[tuple(pt)] = f_nest[i]

    def training_coords(self) -> np.ndarray:
        """ Returns all the coordinates of the points passed to the GP formatted as a numpy array. """
        return np.array([*self._training_pts])

    def training_values(self, scaled: bool = False) -> np.ndarray:
        """ Returns all the values of the points passed to the GP formatted as a numpy array.
            If scaled is True then values scaled by the stored normalisation constants are returned.
            Note that this does not guarantee that the values are normalised. To do so, call rescale() before extracting
            the values.
            If scaled is False then the real values are returned.
        """
        vals = np.array([*self._training_pts.values()]).flatten()
        if scaled:
            return vals
        else:
            return self._denormalise(vals)

    def training_dict(self, scaled: bool = False) -> Dict[np.ndarray, np.ndarray]:
        """ Returns all the point and values passed to the GP as a dictionary.
            If scaled is True then values scaled by the stored normalisation constants are returned.
            Note that this does not guarantee that the values are normalised. To do so, call rescale() before extracting
            the values.
            If scaled is False then the real values are returned.
        """

        if scaled:
            return self._training_pts
        else:
            training_pts = dict(self._training_pts)
            for x in training_pts:
                training_pts[x] = self._denormalise(self._training_pts[x])
            return training_pts

    def sample(self, x: np.ndarray, scaled: bool = False) -> np.ndarray:
        """ Return a sample of the Gaussian Process at the point/s in x.
            If scaled is True then normalised values are returned otherwise they are returned in real space.
        """
        x_nest = np.reshape(x, (-1, self.dims))
        test_count = len(x_nest)

        mean_func, covar_fun = self._calc_kernels(x_nest)
        chol_mat = np.linalg.cholesky(covar_fun + 1e-6 * np.identity(test_count))  # Added for numerical stability
        rand = np.random.normal(0, 1, test_count)

        ans = mean_func.flatten() + np.matmul(chol_mat, rand)
        if not scaled:
            ans = self._denormalise(ans)

        return ans

    def sample_all(self, x: np.ndarray, scaled: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the current mean function and a single standard deviation confidence interval around it at the
            points in x. Returns a tuple in the order (mean, sd).
            If scaled is True then normalised values are returned otherwise they are returned in real space.
        """
        x_nest = np.reshape(x, (-1, self.dims))
        mean_func, covar_fun = self._calc_kernels(x_nest)

        mean_func = mean_func.flatten()
        std = np.sqrt(np.diag(covar_fun))

        if not scaled:
            lower = mean_func - std
            mean_func = self._denormalise(mean_func)
            std = mean_func - self._denormalise(lower)

        return mean_func, std

    def estimate_mean(self, scaled: bool = False) -> Tuple[float, float]:
        """ Returns the estimate and uncertainty of the mean for the GPR.
            If scaled is True then normalised values are returned otherwise they are returned in real space.
        """
        ones = np.ones(len(self._training_pts))
        train_x = tuple([*self._training_pts])  # Tuple used as dict key in the cache decorator
        train_y = np.reshape([*self._training_pts.values()], (-1, 1))

        invK = self._inv_kernel_matrix(train_x, train_x,
                                       self.sigma_noise + 1e-4)  # NB Added here for numerical stability

        calc = np.matmul(ones, invK)
        calc = np.matmul(calc, np.transpose(ones))
        calc = calc ** -1
        mean_uncertainty = calc

        calc = calc * ones
        calc = np.matmul(calc, invK)
        calc = np.matmul(calc, train_y)
        mean_estimate = calc[0]

        if not scaled:
            lower = mean_estimate - mean_uncertainty
            mean_estimate = self._denormalise(mean_estimate)
            mean_uncertainty = mean_estimate - self._denormalise(lower)

        return mean_estimate, mean_uncertainty

    def rescale(self, normalisation_constants: Optional[Tuple[float, float]] = None):
        """
        Changes the scaling of the data in the GPR. If values for mean and st_dev are provided these are used as the new
        parameters. If not the data is scaled by the mean and standard deviation of the data in the GPR dictionary.
        """
        mean_old, st_old = self.normalisation_constants
        gpr_mean = self._denormalise(self.mean) if self.mean else None

        if normalisation_constants:
            mean_new, st_new = normalisation_constants
        else:
            real_pts = np.array([*self._training_pts.values()]) * st_old + mean_old
            mean_new = np.mean(real_pts)
            st_new = np.std(real_pts)

        for pt in self._training_pts:
            old_space = self._training_pts[pt]
            real_space = old_space * st_old + mean_old
            new_space = (real_space - mean_new) / st_new
            self._training_pts[pt] = new_space

        self.normalisation_constants = (mean_new, st_new)
        self.mean = self._normalise(gpr_mean) if self.mean else None

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

    # # BELOW FUNCTION DOES NOT WORK AT EVEN MODERATE DIMENSIONS :(
    # def _inv_kernel_training_matrix(self):
    #     """ Calculates the inverse of the kernel matrix of training points. This is done blockwise according to the
    #     Woodbury matrix identity since the update of the inverse is significantly faster than inverting the entire
    #     matrix from scratch again as it uses the previously calculated inverse of training points. The final matrix
    #     far is cached."""
    #
    #     n_train_pts = len(self._training_pts)
    #     x_train_pts = np.array([*self._training_pts])
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
