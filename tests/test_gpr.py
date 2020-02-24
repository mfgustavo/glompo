

from ..core.gpr import GaussianProcessRegression
from ..core.expkernel import ExpKernel

import numpy as np


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


class TestGPR2D:
    gpr = GaussianProcessRegression(kernel=_SEKernel(),
                                    dims=2,
                                    sigma_noise=0,
                                    mean=None,
                                    cache_results=False)
    for i in np.linspace(0, 2 * np.pi, 10):
        for j in np.linspace(0, 2 * np.pi, 10):
            gpr.add_known((i, j), np.sin(i) * np.sin(j) + 10)

    def test_dict(self):
        loc_gpr = GaussianProcessRegression(kernel=_SEKernel(),
                                            dims=2,
                                            sigma_noise=0,
                                            mean=None,
                                            cache_results=False)
        for i in range(3):
            loc_gpr.add_known((i, i+1), (i+5)**2)

        for i, x in enumerate(loc_gpr.training_pts):
            assert x[0] == i
            assert x[1] == i+1
            assert loc_gpr.training_pts[x] == (i+5)**2

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


class TestGPR1D:
    gpr = GaussianProcessRegression(kernel=ExpKernel(0.1, 5.00),
                                    dims=1,
                                    sigma_noise=0,
                                    mean=None,
                                    cache_results=False)
    for i in range(10):
        gpr.add_known(i, 0.5 * np.exp(- 0.2 * i))

    def test_dict(self):
        for i, x in enumerate(self.gpr.training_pts):
            assert x[0] == i
            assert self.gpr.training_pts[x] == 0.5 * np.exp(- 0.2 * i)

    def test_mean(self):
        before = self.gpr.sample_all(500000)[0]
        self.gpr.mean = 8
        after = self.gpr.sample_all(500000)[0]
        assert not np.isclose(before, after)
        self.gpr.mean = None

    def test_sample_all(self):
        np.random.seed(1)
        x_pts = np.random.rand(25) * 10
        mean, std = self.gpr.sample_all(x_pts)
        assert len(mean) == 25
        assert len(std) == 25
        assert np.ndim(mean) == 1
        assert np.ndim(std) == 1

    def test_dims_sample(self):
        np.random.seed(1)
        x_pts = np.random.rand(5) * 10
        y_pts = self.gpr.sample(x_pts)
        assert len(y_pts) == 5
        assert np.ndim(y_pts) == 1

    def test_estmean(self):
        assert np.isclose(self.gpr.estimate_mean()[0], 5, atol=1e-2)

    def test_noise(self):
        before = self.gpr.sample_all(np.linspace(0.5, 20.5, 20))[1]
        self.gpr.sigma_noise = 0.01
        after = self.gpr.sample_all(np.linspace(0.5, 20.5, 20))[1]
        assert np.all(before < after)
        self.gpr.sigma_noise = 0
