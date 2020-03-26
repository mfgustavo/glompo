

import numpy as np

from glompo.core.gpr import GaussianProcessRegression
from glompo.core.expkernel import ExpKernel


class _SEKernel:
    """ Implements and calculates instances of the squared-exponential covariance function. """

    @staticmethod
    def _norm(x: np.ndarray) -> float:
        return np.sqrt(np.sum(x ** 2))

    def __init__(self, len_scale: float = 1, sigma_signal: float = 1, sigma_noise: float = 0):
        """ Initialises the kernel hyper-parameters.

            Parameters:
            -----------
            len_scale : float
                Length scale hyper-parameter.
            sigma_signal : float
                Standard deviation of the signal.
            sigma_noise : float
                Standard deviation of the noise in given data points.
        """
        self.len_scale = len_scale
        self.sigma_signal = sigma_signal
        self.sigma_noise = sigma_noise

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        calc = self.sigma_signal ** 2
        calc *= np.exp(- 0.5 * self.len_scale ** -2 * self._norm(x1 - x2) ** 2)
        calc += self.sigma_noise ** 2 * np.all(x1 == x2)
        return calc


class TestDictionary:
    loc_gpr = GaussianProcessRegression(kernel=_SEKernel(),
                                        dims=2,
                                        sigma_noise=0,
                                        mean=None,
                                        cache_results=False)
    for i in range(10):
        loc_gpr.add_known((i, i + 1), (i + 5) ** 2)

    def test_hidden_dict(self):
        for i, x in enumerate(self.loc_gpr._training_pts):
            assert x[0] == i
            assert x[1] == i + 1
            assert self.loc_gpr._training_pts[x] == (i + 5) ** 2

    def test_coords_getter(self):
        for i, x in enumerate(self.loc_gpr.training_coords()):
            assert x[0] == i
            assert x[1] == i + 1

    def test_vals_getter(self):
        for i, y in enumerate(self.loc_gpr.training_values()):
            assert y == (i + 5) ** 2

    def test_dict_getter(self):
        tp_dict = self.loc_gpr.training_dict()
        for i, x in enumerate(tp_dict):
            assert x[0] == i
            assert x[1] == i + 1
            assert tp_dict[x] == (i + 5) ** 2

    def test_vals_getter_denormed(self):
        self.loc_gpr.rescale()
        for i, y in enumerate(self.loc_gpr.training_values()):
            assert np.isclose(y, (i + 5) ** 2)
        for i, y in enumerate(self.loc_gpr.training_values(True)):
            assert np.isclose(y, ((i + 5) ** 2 - 98.5) / 55.0549725)

    def test_dict_getter_denormed(self):
        self.loc_gpr.rescale()
        tp_dict = self.loc_gpr.training_dict()
        for i, x in enumerate(tp_dict):
            assert x[0] == i
            assert x[1] == i + 1
            assert np.isclose(tp_dict[x], (i + 5) ** 2)
        tp_dict = self.loc_gpr.training_dict(True)
        for i, x in enumerate(tp_dict):
            assert x[0] == i
            assert x[1] == i + 1
            assert np.isclose(tp_dict[x], ((i + 5) ** 2 - 98.5) / 55.0549725)


class TestGPR2D:
    gpr = GaussianProcessRegression(kernel=_SEKernel(),
                                    dims=2,
                                    sigma_noise=0,
                                    mean=None,
                                    cache_results=True)
    for i in np.linspace(0, 2 * np.pi, 10):
        for j in np.linspace(0, 2 * np.pi, 10):
            gpr.add_known((i, j), np.sin(i) * np.sin(j) + 10)

    def test_sample_all(self):
        np.random.seed(1)
        x_pts = np.random.rand(25, 2) * 2 * np.pi
        mean, std = self.gpr.sample_all(x_pts)
        assert len(mean) == 25
        assert len(std) == 25
        assert np.ndim(mean) == 1
        assert np.ndim(std) == 1

    def test_sample(self):
        np.random.seed(1)
        x_pts = np.random.rand(25, 2) * 2 * np.pi
        y_pts = self.gpr.sample(x_pts)
        assert len(y_pts) == 25
        assert np.ndim(y_pts) == 1

    def test_estmean(self):
        assert np.isclose(self.gpr.estimate_mean()[0], 10)

    def test_regression(self):
        y_est = np.array([])
        y_ref = np.array([])
        for i in np.linspace(0, 2 * np.pi, 10):
            for j in np.linspace(0, 2 * np.pi, 10):
                np.append(y_est, self.gpr.sample_all([i, j])[0])
                np.append(y_ref, [np.sin(i) * np.sin(j) + 10])
        assert np.allclose(y_est, y_ref, rtol=1e-3)

    def test_rescale1(self):
        self.gpr.rescale()
        y = [*self.gpr._training_pts.values()]
        assert np.isclose(np.mean(y), 0)
        assert np.isclose(np.std(y), 1)

    def test_rescale2(self):
        self.gpr.rescale()
        self.gpr.rescale()
        self.gpr.rescale()
        y = [*self.gpr._training_pts.values()]
        assert np.isclose(np.mean(y), 0)
        assert np.isclose(np.std(y), 1)

    def test_rescale3(self):
        self.gpr.rescale((10, 0.45))
        y_new = [*self.gpr._training_pts.values()]
        assert np.isclose(np.mean(y_new), 0)
        assert np.isclose(np.std(y_new), 1)
        assert self.gpr.normalisation_constants == (10, 0.45)

    def test_denorm(self):
        for x in self.gpr._training_pts:
            assert np.sin(x[0]) * np.sin(x[1]) + 10 == self.gpr._denormalise(self.gpr._training_pts[x])


def f(x):
    return 0.5 * np.exp(- 0.2 * x) + 3


class TestGPR1D:
    gpr = GaussianProcessRegression(kernel=ExpKernel(0.1, 5.00),
                                    dims=1,
                                    sigma_noise=0.0001,
                                    mean=None,
                                    cache_results=False)

    for i in range(10):
        gpr.add_known(i, f(i))

    def test_dict(self):
        for i, x in enumerate(self.gpr._training_pts):
            assert x[0] == i
            assert self.gpr._training_pts[x] == f(i)

    def test_mean(self):
        before = self.gpr.sample_all(500000)[0]
        assert np.isclose(before, 3.5, rtol=0.5)
        self.gpr.mean = 8
        after = self.gpr.sample_all(500000)[0]
        assert np.isclose(after, 8, rtol=0.5)
        self.gpr.mean = None

    def test_sample_all(self):
        np.random.seed(1)
        x_pts = np.random.rand(25) * 10
        mean, std = self.gpr.sample_all(x_pts)
        assert len(mean) == 25
        assert len(std) == 25
        assert np.ndim(mean) == 1
        assert np.ndim(std) == 1

    def test_sample(self):
        np.random.seed(1)
        x_pts = np.random.rand(25) * 20
        y_pts = self.gpr.sample(x_pts)
        assert len(y_pts) == 25
        assert np.ndim(y_pts) == 1

    def test_estmean(self):
        assert np.isclose(self.gpr.estimate_mean()[0], 3, rtol=1e-1)

    def test_noise(self):
        before = self.gpr.sample_all(np.linspace(0.5, 20.5, 20))[1]
        self.gpr.sigma_noise = 0.1
        after = self.gpr.sample_all(np.linspace(0.5, 20.5, 20))[1]
        assert np.all(before < after)
        self.gpr.sigma_noise = 0.0001

    def test_regression(self):
        y_est = self.gpr.sample_all(range(10))[0]
        y_ref = np.array([f(i) for i in range(10)])
        assert np.allclose(y_est, y_ref, rtol=1e-3)

    def test_rescale1(self):
        y_old = [*self.gpr._training_pts.values()]
        mean_old = np.mean(y_old)
        st_old = np.std(y_old)

        self.gpr.rescale((mean_old, st_old))
        y_new = [*self.gpr._training_pts.values()]

        assert np.isclose(np.mean(y_new), 0)
        assert np.isclose(np.std(y_new), 1)
        assert self.gpr.normalisation_constants == (mean_old, st_old)

    def test_rescale2(self):
        self.gpr.rescale()
        y = [*self.gpr._training_pts.values()]
        assert np.isclose(np.mean(y), 0)
        assert np.isclose(np.std(y), 1)

    def test_rescale3(self):
        self.gpr.rescale()
        self.gpr.rescale()
        self.gpr.rescale()
        y = [*self.gpr._training_pts.values()]
        assert np.isclose(np.mean(y), 0)
        assert np.isclose(np.std(y), 1)

    def test_denorm(self):
        for i, x in enumerate(self.gpr._denormalise(np.array([*self.gpr._training_pts.values()]))):
            assert f(i) == x
