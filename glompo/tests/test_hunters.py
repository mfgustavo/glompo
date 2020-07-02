from typing import Callable, Sequence, Tuple

import numpy as np
import pytest

from glompo.common.corebase import _CombiCore
from glompo.core.optimizerlogger import OptimizerLogger
from glompo.hunters.basehunter import BaseHunter, _AndHunter, _OrHunter
from glompo.hunters.lastptsinvalid import LastPointsInvalid
from glompo.hunters.min_fcalls import MinFuncCalls
from glompo.hunters.min_iterations import MinIterations
from glompo.hunters.parameterdistance import ParameterDistance
from glompo.hunters.pseudoconv import PseudoConverged
from glompo.hunters.timeannealing import TimeAnnealing
from glompo.hunters.type import TypeHunter
from glompo.hunters.valueannealing import ValueAnnealing
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult


class PlainHunter(BaseHunter):
    def __call__(self, log, hunter_opt_id, victim_opt_id) -> bool:
        pass


class TrueHunter(BaseHunter):
    def __call__(self, log, hunter_opt_id, victim_opt_id) -> bool:
        return True


class FalseHunter(BaseHunter):
    def __call__(self, log, hunter_opt_id, victim_opt_id) -> bool:
        return False


class FancyHunter(BaseHunter):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b + c

    def __call__(self, log, hunter_opt_id, victim_opt_id) -> bool:
        pass


def any_hunter():
    return _OrHunter(PlainHunter(), PlainHunter())


def all_hunter():
    return _AndHunter(PlainHunter(), PlainHunter())


class FakeLog:
    def __init__(self, *args):
        self.path = [*args]

    def __len__(self):
        return len(self.path)

    def get_history(self, opt_id, track):
        if track != "f_call_opt":
            return self.path[opt_id - 1]
        return list(range(1, len(self.path[opt_id - 1]) + 1))

    @staticmethod
    def get_metadata(*args):
        return {2: "FakeOpt", 8: "XXXOpt"}[args[0]]


class FakeOpt(BaseOptimizer):

    def minimize(self, function: Callable[[Sequence[float]], float], x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]], callbacks: Callable = None, **kwargs) -> MinimizeResult:
        pass

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def save_state(self, *args):
        pass


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
        assert hunter(*(None,) * 3) is True


class TestMinTrainingPoints:
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
        assert cond(log, None, None) == output

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
            log.put_iteration(1, i, calls_per_iter * i, calls_per_iter * i, i, 10)
        for i in range(10, 20):
            log.put_iteration(1, i, calls_per_iter * i, calls_per_iter * i, i, 1)
        for i in range(20, 30):
            log.put_iteration(1, i, calls_per_iter * i, calls_per_iter * i, i, 0.9)
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
        assert cond(log, None, 1) is output


class TestParameterDistance:

    @pytest.mark.parametrize("paths, bounds, rel_dist, test_all, output", [(([[0, 0], [0, 1], [0, 2]],
                                                                             [[1, 0], [1, 1], [1, 2]]),
                                                                            [(0, 2)] * 3, 0.1, False, False),
                                                                           (([[0, 0], [0, 1], [0, 2]],
                                                                             [[1, 0], [1, 1], [1, 2]]),
                                                                            [(0, 2)] * 3, 0.5, False, True),
                                                                           (([[0, 0], [0, 1], [1, 2]],
                                                                             [[1, 0], [1, 1], [1, 2]]),
                                                                            [(0, 2)] * 3, 0.1, False, True),
                                                                           (([[0, 0], [10, 10], [20, 20]],
                                                                             [[20, 18], [20, 19], [20, 21]]),
                                                                            [(0, 20)] * 3, 0.1, False, True),
                                                                           (([[0, 0], [0, 0.1], [0, 0.2]],
                                                                             [[0, 0], [10, 10], [0, 0.25]]),
                                                                            [(0, 10)] * 3, 0.1, False, True),
                                                                           (([[0, 0], [100, 100], [0, 1]],
                                                                             [[1, 0], [1, 1], [0, 1.1]]),
                                                                            [(0, 100)] * 3, 0.11, False, True),
                                                                           (([[0, 0], [0, 1], [0, 2]],
                                                                             [[1, 0], [1, 1], [1, 2]],
                                                                             [],
                                                                             [],
                                                                             []),
                                                                            [(0, 2)] * 3, 0.5, True, True),
                                                                           (([[0, 0], [0, 1], [0, 2]],
                                                                             [[0, 0], [0, 1], [1, 2]],
                                                                             [[0, 0], [0, 1], [0, 3]],
                                                                             [[1, 0], [1, 1], [1.3, 2]],
                                                                             [[0, 0], [0, 1], [4, 2]]),
                                                                            [(0, 2)] * 3, 0.1, True, True),
                                                                           (([[0, 0], [0, 1], [0, 2]],
                                                                             [[0, 0], [0, 1], [1, 2]],
                                                                             [[0, 0], [0, 1], [0, 3]],
                                                                             [[1, 0], [1, 1], [0.3, 2]],
                                                                             [[0, 0], [0, 1], [4, 2]]),
                                                                            [(0, 2)] * 3, 0.1, True, False)
                                                                           ])
    def test_condition(self, paths, bounds, rel_dist, test_all, output):
        cond = ParameterDistance(bounds, rel_dist, test_all)
        log = FakeLog(*paths)
        assert cond(log, 1, 2) == output

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
        assert cond(log, 1, 2) == output

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
        assert cond(log, 1, 2) == output


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
        assert cond(log, 1, 2) == output


class TestMinFCalls:

    @pytest.mark.parametrize("path, output",
                             [([12, 12, 12, 12, 12], True),
                              ([1, 1, 1], True),
                              ([3, 3], False)])
    def test_condition(self, path, output):
        cond = MinFuncCalls(3)
        log = FakeLog([], path)
        assert cond(log, 1, 2) == output


class TestTypeHunter:

    @pytest.mark.parametrize("vic, output",
                             [(2, True),
                              (8, False)])
    def test_condition(self, vic, output):
        cond = TypeHunter(FakeOpt)
        log = FakeLog([], [])
        assert cond(log, 1, vic) == output
