

import pytest
import numpy as np

from glompo.core.regression import DataRegressor


class TestRegressor:

    t = np.arange(0, 10)
    y = (756 * np.exp(- 0.235 * t) + 200) + np.random.uniform(-1, 1, len(t))

    reg = DataRegressor()

    def assert_mle_results(self, mle):
        b, c, s = mle
        assert np.isclose(b, 0.235, atol=0.005)
        assert np.isclose(c, 200/self.y[-1], atol=0.01)
        assert np.isclose(s, 1/self.y[0], atol=1e-3)

    def assert_mcmc_results(self, mcmc):
        b, c, s = mcmc
        b_med, b_low, b_upp = b
        c_med, c_low, c_upp = c
        s_med, s_low, s_upp = s

        assert np.isclose(b_med, 0.235, atol=0.005)
        assert 0 < b_low < b_med < b_upp

        assert np.isclose(c_med, 200 / self.y[-1], atol=0.01)
        assert 0 < c_low < c_med < c_upp < 1

        assert np.isclose(s_med, 1 / self.y[0], atol=1e-3)
        assert 0 < s_low < s_med < s_upp

    def test_mle(self):
        mle = self.reg.estimate_mle(self.t, self.y, cache_key=1)
        self.assert_mle_results(mle)

    def test_mcmc(self):
        mcmc = self.reg.estimate_posterior(self.t, self.y, nwalkers=10, nsteps=3000, cache_key=1)
        self.assert_mcmc_results(mcmc)

    def test_cache(self):

        self.reg.log_like = None
        self.reg.log_post = None

        mle = self.reg.estimate_mle(self.t, self.y, cache_key=1)
        self.assert_mle_results(mle)
        with pytest.raises(Exception):
            self.reg.estimate_mle(self.t, self.y)

        mcmc = self.reg.estimate_posterior(self.t, self.y, cache_key=1)
        self.assert_mcmc_results(mcmc)
        with pytest.raises(Exception):
            self.reg.estimate_posterior(self.t, self.y)

    def test_get_methods(self):

        mle = self.reg.get_mle_results(1)
        self.assert_mle_results(mle)

        mcmc = self.reg.get_mcmc_results(1)
        self.assert_mcmc_results(mcmc)
