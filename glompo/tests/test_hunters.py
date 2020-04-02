

import pytest
import numpy as np

from glompo.hunters.basehunter import BaseHunter, _AllHunter, _AnyHunter, _CombiHunter
from glompo.hunters.confidencewidth import ConfidenceWidth
from glompo.hunters.min_vic_trainpts import MinVictimTrainingPoints
from glompo.hunters.val_below_asymptote import ValBelowGPR
from glompo.hunters.pseudoconv import PseudoConverged
from glompo.core.logger import Logger


class PlainHunter(BaseHunter):
    def __init__(self):
        pass

    def is_kill_condition_met(self, log, hunter_opt_id, victim_opt_id) -> bool:
        pass


class TrueHunter(BaseHunter):
    def is_kill_condition_met(self, log, hunter_opt_id, victim_opt_id) -> bool:
        return True


class FalseHunter(BaseHunter):
    def is_kill_condition_met(self, log, hunter_opt_id, victim_opt_id) -> bool:
        return False


class FancyHunter(BaseHunter):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b + c

    def is_kill_condition_met(self, log, hunter_opt_id, victim_opt_id) -> bool:
        pass


def any_hunter():
    return _AnyHunter(PlainHunter(), PlainHunter())


def all_hunter():
    return _AllHunter(PlainHunter(), PlainHunter())


class TestBase:
    @pytest.mark.parametrize("base1, base2", [(PlainHunter(), PlainHunter()),
                                              (PlainHunter(), any_hunter()),
                                              (any_hunter(), PlainHunter()),
                                              (PlainHunter(), all_hunter()),
                                              (all_hunter(), PlainHunter()),
                                              (any_hunter(), all_hunter())])
    def test_or(self, base1, base2):
        assert (base1 | base2).__class__.__name__ == "_AnyHunter"

    @pytest.mark.parametrize("base1, base2", [(PlainHunter(), PlainHunter()),
                                              (PlainHunter(), any_hunter()),
                                              (any_hunter(), PlainHunter()),
                                              (PlainHunter(), all_hunter()),
                                              (all_hunter(), PlainHunter()),
                                              (any_hunter(), all_hunter())])
    def test_and(self, base1, base2):
        assert (base1 & base2).__class__.__name__ == "_AllHunter"

    @pytest.mark.parametrize("hunter, output", [(PlainHunter(), "PlainHunter()"),
                                                (any_hunter(), "PlainHunter() OR \nPlainHunter()"),
                                                (all_hunter(), "PlainHunter() AND \nPlainHunter()"),
                                                (FancyHunter(1, 2, 3), "FancyHunter(a=1, b=5, c)")])
    def test_str(self, hunter, output):
        assert str(hunter) == output

    def test_combi_init(self):
        with pytest.raises(TypeError):
            _CombiHunter(1, 2, 3)

    def test_kill_condition(self):
        hunter = FalseHunter() | FalseHunter() & TrueHunter() | TrueHunter() & (TrueHunter() | FalseHunter())
        assert hunter.is_kill_condition_met(*(None,) * 3) is True


# class TestConfidenceWidth:
#
#     @pytest.mark.parametrize("threshold, output", [(0.01, False),
#                                                    (0.13, True),
#                                                    (0.125, False),
#                                                    (0.1251, True),
#                                                    (2.3, True)])
#     def test_condition(self, threshold, output):
#         cond = ConfidenceWidth(threshold)
#         assert cond.is_kill_condition_met(None, None, None) == output
#
#     @pytest.mark.parametrize("threshold", [-5, -5.0, 0])
#     def test_init_crash(self, threshold):
#         with pytest.raises(ValueError):
#             ConfidenceWidth(threshold)
#
#     @pytest.mark.parametrize("threshold", [5, 5.0, 0.015, 0.863])
#     def test_init_pass(self, threshold):
#         ConfidenceWidth(threshold)


# class TestMinTraningPoints:
#     @pytest.mark.parametrize("n_pts, output", [(1, False),
#                                                (2, False),
#                                                (3, False),
#                                                (5, True),
#                                                (6, True)])
#     def test_condition(self, n_pts, output):
#         cond = MinVictimTrainingPoints(5)
#         vic_gpr = GaussianProcessRegression(kernel=None,
#                                             dims=1)
#         for _ in range(n_pts):
#             vic_gpr.add_known(np.random.rand(1), np.random.rand(1))
#         assert cond.is_kill_condition_met(None, None, None, None, vic_gpr) == output
#
#     @pytest.mark.parametrize("threshold", [-5, -5.0, 0, 32.5, 0.25])
#     def test_init_crash(self, threshold):
#         with pytest.raises(ValueError):
#             MinVictimTrainingPoints(threshold)
#
#     @pytest.mark.parametrize("threshold", [5, 2])
#     def test_init_pass(self, threshold):
#         MinVictimTrainingPoints(threshold)


# class TestValBelowGPR:
#
#     class FakeGPR:
#         @staticmethod
#         def estimate_mean():
#             return 800.0, 100.0
#
#     @pytest.mark.parametrize("const, output", [(0, True),
#                                                (600, True),
#                                                (601, False),
#                                                (700, False),
#                                                (1000, False)])
#     def test_condition(self, const, output):
#         cond = ValBelowGPR()
#         hunter_opt_id = 1
#
#         log = Logger()
#         log.add_optimizer(hunter_opt_id, None, None)
#         count = 0
#         for i in np.linspace(0, 2*np.pi, 10):
#             count += 1
#             log.put_iteration(hunter_opt_id, count, count, i, np.sin(i) + const)
#
#         assert cond.is_kill_condition_met(log, hunter_opt_id, None, None, self.FakeGPR()) == output
#
#     def test_no_history(self):
#         cond = ValBelowGPR()
#         hunter_opt_id = 1
#
#         log = Logger()
#         log.add_optimizer(hunter_opt_id, None, None)
#
#         assert cond.is_kill_condition_met(log, hunter_opt_id, None, None, self.FakeGPR()) is False


class TestPseudoConv:
    @pytest.fixture()
    def log(self):
        log = Logger()
        log.add_optimizer(1, None, None)
        for i in range(10):
            log.put_iteration(1, i, i, i, 10)
        for i in range(10, 20):
            log.put_iteration(1, i, i, i, 1)
        for i in range(20, 30):
            log.put_iteration(1, i, i, i, 0.9)
        return log

    @pytest.mark.parametrize("iters, tol, output", [(10, 0, True),
                                                    (8, 0, True),
                                                    (11, 0, False),
                                                    (11, 0.1, True),
                                                    (20, 0.1, True),
                                                    (60, 0, False),
                                                    (60, 0.90, False),
                                                    (25, 0.91, True)])
    def test_condition(self, iters, tol, output, log):
        cond = PseudoConverged(iters, tol)
        assert cond.is_kill_condition_met(log, None, 1) is output


# class TestGPRSuitable:
#     @pytest.mark.parametrize("noise1, noise2, output", [(1, 1, False),
#                                                         (0.005, 0.005, True),
#                                                         (0.005, 1, False),
#                                                         (1, 0.005, False)])
#     def test_cond(self, noise1, noise2, output):
#         np.random.seed(1)
#         kernel = ExpKernel(alpha=0.7,
#                            beta=0.1)
#         log = Logger()
#         log.add_optimizer(1, None, None)
#         log.add_optimizer(2, None, None)
#         gpr1 = GaussianProcessRegression(kernel=kernel,
#                                          dims=1,
#                                          sigma_noise=noise1)
#         gpr2 = GaussianProcessRegression(kernel=kernel,
#                                          dims=1,
#                                          sigma_noise=noise2)
#
#         t_short = np.arange(0, 10)
#         pts = np.exp(-t_short) * np.random.uniform(0.5, 1.5, len(t_short)) + 0.1
#         for x, y in enumerate(pts):
#             gpr1.add_known(x, y)
#             gpr2.add_known(x, y)
#             log.put_iteration(1, x, x, x, y)
#             log.put_iteration(2, x, x, x, y)
#
#         cond = GPRSuitable(0.1)
#
#         assert cond.is_kill_condition_met(log, 1, gpr1, 2, gpr2) is output
#
#         if noise1 == 1 and noise2 == 0.001:
#             sigma = kernel.optimize_hyperparameters(t_short, pts)[2]
#             gpr1.sigma_noise = sigma
#
#             assert cond.is_kill_condition_met(log, 1, gpr1, 2, gpr2) is True
#
#     def test_wild_tail(self):
#         kernel = ExpKernel(alpha=1.2,
#                            beta=10)
#         gpr = GaussianProcessRegression(kernel=kernel,
#                                         dims=1,
#                                         sigma_noise=0.00001)
#         log = Logger()
#         log.add_optimizer(1, None, None)
#         log.add_optimizer(2, None, None)
#
#         t = np.arange(0, 10)
#         pts = np.exp(-t) * np.random.uniform(0.5, 1.5, len(t))
#         for x, y in enumerate(pts):
#             gpr.add_known(x, y)
#             log.put_iteration(1, x, x, x, y)
#             log.put_iteration(2, x, x, x, y)
#
#         cond = GPRSuitable(0.03)
#
#         assert cond.is_kill_condition_met(log, 1, gpr, 2, gpr) is False
