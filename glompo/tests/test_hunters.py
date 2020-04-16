

import pytest
import numpy as np

from glompo.hunters.basehunter import BaseHunter, _AllHunter, _AnyHunter, _CombiHunter
from glompo.hunters.confidencewidth import ConfidenceWidth
from glompo.hunters.min_iterations import MinIterations
from glompo.hunters.val_below_asymptote import ValBelowAsymptote
from glompo.hunters.pseudoconv import PseudoConverged
from glompo.core.logger import Logger


class PlainHunter(BaseHunter):
    def __init__(self):
        pass

    def is_kill_condition_met(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
        pass


class TrueHunter(BaseHunter):
    def is_kill_condition_met(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
        return True


class FalseHunter(BaseHunter):
    def is_kill_condition_met(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
        return False


class FancyHunter(BaseHunter):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b + c

    def is_kill_condition_met(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
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
            _CombiHunter(1, 2)

    def test_kill_condition(self):
        hunter = FalseHunter() | FalseHunter() & TrueHunter() | TrueHunter() & (TrueHunter() | FalseHunter())
        assert hunter.is_kill_condition_met(*(None,) * 4) is True


class TestConfidenceWidth:

    class FakeRegressor:
        @staticmethod
        def get_mcmc_results(*args):
            return 800.0, 750.0, 850.0

    @pytest.mark.parametrize("threshold, output", [(0.01, False),
                                                   (0.13, True),
                                                   (0.125, False),
                                                   (0.1251, True),
                                                   (2.3, True)])
    def test_condition(self, threshold, output):
        cond = ConfidenceWidth(threshold)
        assert cond.is_kill_condition_met(None, self.FakeRegressor(), None, None) == output

    @pytest.mark.parametrize("threshold", [-5, -5.0, 0])
    def test_init_crash(self, threshold):
        with pytest.raises(ValueError):
            ConfidenceWidth(threshold)

    @pytest.mark.parametrize("threshold", [5, 5.0, 0.015, 0.863])
    def test_init_pass(self, threshold):
        ConfidenceWidth(threshold)


class TestMinTraningPoints:

    class FakeLog:
        def __init__(self, n_pts):
            self.history = np.ones(n_pts)

        def get_history(self, *args):
            return self.history

    @pytest.mark.parametrize("n_pts, output", [(1, False),
                                               (2, False),
                                               (3, False),
                                               (5, True),
                                               (6, True)])
    def test_condition(self, n_pts, output):
        cond = MinIterations(5)
        log = self.FakeLog(n_pts)
        assert cond.is_kill_condition_met(log, None, None, None) == output

    @pytest.mark.parametrize("threshold", [-5, -5.0, 0, 32.5, 0.25])
    def test_init_crash(self, threshold):
        with pytest.raises(ValueError):
            MinIterations(threshold)

    @pytest.mark.parametrize("threshold", [5, 2])
    def test_init_pass(self, threshold):
        MinIterations(threshold)


class TestPseudoConv:
    @pytest.fixture
    def log(self, request):
        log = Logger()
        log.add_optimizer(1, None, None)
        calls_per_iter = request.param
        for i in range(10):
            log.put_iteration(1, i, calls_per_iter*i, i, 10)
        for i in range(10, 20):
            log.put_iteration(1, i, calls_per_iter*i, i, 1)
        for i in range(20, 30):
            log.put_iteration(1, i, calls_per_iter*i, i, 0.9)
        return log

    @pytest.mark.parametrize("iters, tol, output, log", [(10, 0, False, 1),
                                                         (8, 0, True, 1),
                                                         (11, 0, False, 1),
                                                         (11, 0.1, True, 1),
                                                         (20, 0.1, False, 1),
                                                         (60, 0, False, 1),
                                                         (60, 0.90, False, 1),
                                                         (25, 0.91, True, 1),
                                                         (30, 0, False, 3),
                                                         (125, 0.91, True, 5)], indirect=("log",))
    def test_condition(self, iters, tol, output, log):
        cond = PseudoConverged(iters, tol)
        assert cond.is_kill_condition_met(log, None, None, 1) is output
