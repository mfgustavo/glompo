

import pytest
import numpy as np

from glompo.common.corebase import _CombiCore
from glompo.hunters.basehunter import BaseHunter, _OrHunter, _AndHunter
from glompo.hunters.confidencewidth import ConfidenceWidth
from glompo.hunters.min_iterations import MinIterations
from glompo.hunters.pseudoconv import PseudoConverged
from glompo.hunters.parameterdistance import ParameterDistance
from glompo.hunters.timeannealing import TimeAnnealing
from glompo.hunters.valueannealing import ValueAnnealing
from glompo.hunters.val_below_asymptote import ValBelowAsymptote
from glompo.hunters.lastptsinvalid import LastPointsInvalid
from glompo.core.optimizerlogger import OptimizerLogger


class PlainHunter(BaseHunter):
    def __call__(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
        pass


class TrueHunter(BaseHunter):
    def __call__(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
        return True


class FalseHunter(BaseHunter):
    def __call__(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
        return False


class FancyHunter(BaseHunter):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b + c

    def __call__(self, log, regressor, hunter_opt_id, victim_opt_id) -> bool:
        pass


def any_hunter():
    return _OrHunter(PlainHunter(), PlainHunter())


def all_hunter():
    return _AndHunter(PlainHunter(), PlainHunter())


class FakeLog:
    def __init__(self, path1, path2):
        self.path = [path1, path2]

    def get_history(self, opt_id, track):
        if track != "f_call_opt":
            return self.path[opt_id - 1]
        return list(range(1, len(self.path[opt_id-1])+1))


class TestBase:
    @pytest.mark.parametrize("base1, base2", [(PlainHunter(), PlainHunter()),
                                              (PlainHunter(), any_hunter()),
                                              (any_hunter(), PlainHunter()),
                                              (PlainHunter(), all_hunter()),
                                              (all_hunter(), PlainHunter()),
                                              (any_hunter(), all_hunter())])
    def test_or(self, base1, base2):
        assert (base1 | base2).__class__.__name__ == "_OrHunter"

    @pytest.mark.parametrize("base1, base2", [(PlainHunter(), PlainHunter()),
                                              (PlainHunter(), any_hunter()),
                                              (any_hunter(), PlainHunter()),
                                              (PlainHunter(), all_hunter()),
                                              (all_hunter(), PlainHunter()),
                                              (any_hunter(), all_hunter())])
    def test_and(self, base1, base2):
        assert (base1 & base2).__class__.__name__ == "_AndHunter"

    @pytest.mark.parametrize("hunter, output", [(PlainHunter(), "PlainHunter()"),
                                                (any_hunter(), "[PlainHunter() | \nPlainHunter()]"),
                                                (all_hunter(), "[PlainHunter() & \nPlainHunter()]"),
                                                (FancyHunter(1, 2, 3), "FancyHunter(a=1, b=5, c)")])
    def test_str(self, hunter, output):
        assert str(hunter) == output

    def test_combi_init(self):
        with pytest.raises(TypeError):
            _CombiCore(1, 2)

    def test_kill_condition(self):
        hunter = FalseHunter() | FalseHunter() & TrueHunter() | TrueHunter() & (TrueHunter() | FalseHunter())
        assert hunter(*(None,) * 4) is True


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
        assert cond(None, self.FakeRegressor(), None, None) == output

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
        assert cond(log, None, None, None) == output

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
        log = OptimizerLogger()
        log.add_optimizer(1, None, None)
        calls_per_iter = request.param
        for i in range(10):
            log.put_iteration(1, i, calls_per_iter*i, calls_per_iter*i, i, 10)
        for i in range(10, 20):
            log.put_iteration(1, i, calls_per_iter*i, calls_per_iter*i, i, 1)
        for i in range(20, 30):
            log.put_iteration(1, i, calls_per_iter*i, calls_per_iter*i, i, 0.9)
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
        assert cond(log, None, None, 1) is output


class TestParameterDistance:

    @pytest.mark.parametrize("path1, path2, bounds, rel_dist, output", [([[0, 0], [0, 1], [0, 2]],
                                                                         [[1, 0], [1, 1], [1, 2]],
                                                                         [(0, 2)] * 3, 0.1, False),
                                                                        ([[0, 0], [0, 1], [0, 2]],
                                                                         [[1, 0], [1, 1], [1, 2]],
                                                                         [(0, 2)] * 3, 0.5, True),
                                                                        ([[0, 0], [0, 1], [1, 2]],
                                                                         [[1, 0], [1, 1], [1, 2]],
                                                                         [(0, 2)] * 3, 0.1, True),
                                                                        ([[0, 0], [10, 10], [20, 20]],
                                                                         [[20, 18], [20, 19], [20, 21]],
                                                                         [(0, 20)] * 3, 0.1, True),
                                                                        ([[0, 0], [0, 0.1], [0, 0.2]],
                                                                         [[0, 0], [10, 10], [0, 0.25]],
                                                                         [(0, 10)] * 3, 0.1, True),
                                                                        ([[0, 0], [100, 100], [0, 1]],
                                                                         [[1, 0], [1, 1], [0, 1.1]],
                                                                         [(0, 100)] * 3, 0.11, True)
                                                                        ])
    def test_condition(self, path1, path2, bounds, rel_dist, output):
        cond = ParameterDistance(bounds, rel_dist)
        log = FakeLog(path1, path2)
        assert cond(log, None, 1, 2) == output

    @pytest.mark.parametrize("rel_dist", [-5, -5.0, 0])
    def test_init_crash(self, rel_dist):
        with pytest.raises(ValueError):
            ParameterDistance([(0, 2)] * 3, rel_dist)

    @pytest.mark.parametrize("rel_dist", [5, 2, 0.2])
    def test_init_pass(self, rel_dist):
        ParameterDistance([(0, 2)] * 3, rel_dist)


class TestTimeAnnealing:

    @pytest.mark.parametrize("path1, path2, crit_ratio, output", [(np.zeros(10), np.zeros(99), 0.1, False),
                                                                  (np.zeros(10), np.zeros(49), 0.2, False),
                                                                  (np.zeros(10), np.zeros(19), 0.5, False),
                                                                  (np.zeros(10), np.zeros(10), 1.0, False),
                                                                  (np.zeros(10), np.zeros(4), 2.0, False),
                                                                  (np.zeros(10), np.zeros(1), 5.0, False)
                                                                  ])
    def test_condition(self, path1, path2, crit_ratio, output):
        np.random.seed(1825)
        cond = TimeAnnealing(crit_ratio)
        log = FakeLog(path1, path2)
        assert cond(log, None, 1, 2) == output

    @pytest.mark.parametrize("rel_dist", [-5, -5.0, 0])
    def test_init_crash(self, rel_dist):
        with pytest.raises(ValueError):
            TimeAnnealing(rel_dist)

    @pytest.mark.parametrize("rel_dist", [5, 2, 0.2])
    def test_init_pass(self, rel_dist):
        TimeAnnealing(rel_dist)


class TestValueAnnealing:

    @pytest.mark.parametrize("path1, path2, output", [([1000], [1], False),
                                                      ([1000], [10], False),
                                                      ([1000], [100], False),
                                                      ([1000], [500], False),
                                                      ([1000], [999], False)
                                                      ])
    def test_condition(self, path1, path2, output):
        cond = ValueAnnealing()
        log = FakeLog(path1, path2)
        assert cond(log, None, 1, 2) == output


class TestValBelowAsymptote:

    class FakeRegressor:
        def __init__(self, result):
            self.result = result

        def estimate_posterior(self, *args, **kwargs):
            return self.result

    @pytest.mark.parametrize("path1, path2, result, output", [(np.full(10, 100), np.ones(10), (105, 97, 120), False),
                                                              (np.full(10, 100), np.ones(10), (99, 99, 101), False),
                                                              (np.full(10, 100), np.ones(10), (5, 2, 5.5), False),
                                                              (np.full(10, 100), np.ones(10), (5,), False),
                                                              (np.full(0, 100),  np.ones(10), (5, 2, 5.5), False),
                                                              (np.full(10, 100), np.ones(10), (101, 100.1, 101), True),
                                                              (np.full(10, 100), np.ones(10), (200, 170, 200), True)
                                                              ])
    def test_condition(self, path1, path2, result, output):
        cond = ValBelowAsymptote()
        log = FakeLog(path1, path2)
        reg = self.FakeRegressor(result)
        assert cond(log, reg, 1, 2) == output


class TestLastPointsInvalid:

    @pytest.mark.parametrize("path, output", [([12, np.inf, np.inf, np.inf, np.inf], False),
                                              ([np.inf, np.inf, np.inf, np.inf], False),
                                              ([np.inf, np.inf, np.inf, np.inf, np.inf], True),
                                              ([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], True),
                                              ([np.inf, np.inf, np.inf, 8, np.inf], False),
                                              ([84, np.inf, np.inf, np.inf, np.inf, np.inf], True),
                                              ([84, 654, np.inf, np.inf, np.inf, np.inf], False)
                                              ])
    def test_condition(self, path, output):
        cond = LastPointsInvalid(5)
        log = FakeLog([], path)
        assert cond(log, None, 1, 2) == output
