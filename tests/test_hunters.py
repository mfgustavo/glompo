# noinspection PyTypeChecker
from typing import Callable, Sequence, Tuple

import numpy as np
import pytest

from glompo.common.corebase import _CombiCore
from glompo.core.optimizerlogger import BaseLogger
from glompo.hunters.basehunter import BaseHunter, _AndHunter, _OrHunter
from glompo.hunters.bestunmoving import BestUnmoving
from glompo.hunters.evalsunmoving import EvaluationsUnmoving
from glompo.hunters.lastptsinvalid import LastPointsInvalid
from glompo.hunters.minfcalls import MinFuncCalls
from glompo.hunters.parameterdistance import ParameterDistance
from glompo.hunters.stepsize import StepSize
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


class FakeLog(BaseLogger):
    def __init__(self, *args):
        self.path = [*args]

    @property
    def n_optimizers(self):
        return len(self.path)

    def get_best_iter(self, opt_id=None):
        return {'opt_id': opt_id,
                'x': [],
                'fx': min(self.path[opt_id - 1]),
                'type': '',
                'call_id': 0}

    def len(self, opt_id):
        return len(self.path[opt_id - 1])

    def get_history(self, opt_id, track):
        return self.path[opt_id - 1]

    def get_metadata(self, *args):
        return {2: "FakeOpt", 8: "XXXOpt"}[args[0]]


class FakeOpt(BaseOptimizer):
    def minimize(self, function: Callable[[Sequence[float]], float], x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]], callbacks: Callable = None, **kwargs) -> MinimizeResult:
        pass

    def callstop(self, *args):
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


class TestBestUnmoving:
    @pytest.mark.parametrize("iters, tol, output", [(12, 0, False),
                                                    (8, 0, True),
                                                    (11, 0, False),
                                                    (11, 0.1, True),
                                                    (20, 0.1, False),
                                                    (60, 0, False),
                                                    (60, 0.90, False),
                                                    (12, 0.91, True),
                                                    (30, 0, False)])
    def test_condition(self, iters, tol, output):
        log = FakeLog([10] * 10 + [1] * 10 + [0.9] * 10)
        cond = BestUnmoving(iters, tol)
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
                                                      ([1000], [999], False),
                                                      ([1000], [2000], False),
                                                      ([1000], [3400], True),
                                                      ([-50], [120], True)])
    def test_condition(self, path1, path2, output):
        np.random.seed(86)
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
                             [([12] * 5, True),
                              ([1] * 3, True),
                              ([3] * 2, False)])
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


class TestStepSize:
    @pytest.fixture()
    def log(self):
        np.random.seed(64)
        history = {1: {'f_call_opt': [*range(1, 201)], 'x': np.random.random((200, 2))}}

        class FakeLog:
            def __init__(self, hist):
                self.hist = hist

            def get_history(self, opt_id, track):
                return self.hist[opt_id][track]

        return FakeLog(history)

    @pytest.mark.parametrize('bnds, calls, tol, output', [([(0, 10)] * 2, 100, 0.1, True),
                                                          ([(0, 10)] * 2, 100, 0.01, False),
                                                          ([(0, 1)] * 2, 100, 0.1, False),
                                                          ([(0, 10)] * 2, 200, 0.1, True),
                                                          ([(0, 10)] * 2, 900, 0.1, False),
                                                          ([(0, 10)] * 2, 3, 0.1, True)])
    def test_condition(self, log, bnds, calls, tol, output):
        cond = StepSize(bnds, calls, tol)
        assert cond(log, None, 1) == output


class TestEvaluationsUnmoving:

    @pytest.fixture()
    def log(self):
        np.random.seed(35)
        history = {1: {'fx': [1 / i * np.random.random() + 1 for i in range(1, 201)]},
                   2: {'fx': [float('inf')] * 10}}

        class FakeLog:
            def __init__(self, hist):
                self.hist = hist

            def len(self, opt_id):
                return len(self.hist[opt_id]['fx'])

            def get_history(self, opt_id, track):
                return self.hist[opt_id][track]

        return FakeLog(history)

    @pytest.mark.parametrize('calls, tol, opt_id, output', [(900, 0.1, 1, False),
                                                            (100, 0.01, 1, True),
                                                            (180, 0.00001, 1, False),
                                                            (3, 0.01, 2, False)])
    def test_condition(self, log, calls, tol, opt_id, output):
        cond = EvaluationsUnmoving(calls, tol)
        np.seterr(invalid='warn')
        warn = None if opt_id == 1 else RuntimeWarning
        with pytest.warns(warn):
            assert cond(log, None, opt_id) == output
