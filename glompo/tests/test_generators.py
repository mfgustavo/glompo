import numpy as np
import pytest

from glompo.common.namedtuples import Bound, Result
from glompo.generators.best import IncumbentGenerator
from glompo.generators.exploit_explore import ExploitExploreGenerator
from glompo.generators.random import RandomGenerator
from glompo.generators.single import SinglePointGenerator

try:
    from glompo.generators.peterbation import PerturbationGenerator

    has_scipy = True
except ModuleNotFoundError:
    has_scipy = False


class TestRandom:

    @pytest.mark.parametrize("bounds", [[[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6, 2e6]],
                                        [Bound(198.5, 200), Bound(0, 6), Bound(-0.00001, 0.001),
                                         Bound(-64.56, -54.54), Bound(1e6, 2e6)],
                                        ((198.5, 200), (0, 6), (-0.00001, 0.001), (-64.56, -54.54), (1e6, 2e6)),
                                        np.array([[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6,
                                                                                                              2e6]]),
                                        ])
    def test_bounds(self, bounds):
        np.random.seed(1)
        gen = RandomGenerator(bounds)

        for _ in range(50):
            call = gen.generate(None)
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    @pytest.mark.parametrize("bounds", [np.array([[1098.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54],
                                                  [1e6, 2e6]]),
                                        np.array([[198.5, 200], [10, 6], [-0.00001, 0.001], [-64.56, -54.54],
                                                  [1e6, 2e6]]),
                                        np.array([[198.5, 200], [0, 6], [0.00001, -0.001], [-64.56, -54.54],
                                                  [1e6, 2e6]]),
                                        np.array([[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -64.56],
                                                  [1e6, 2e6]]),
                                        np.array([[-np.inf, 200], [0, 6], [-0.00001, 0.001], [-64.56, -64.56],
                                                  [1e6, 2e6]])])
    def test_invalid_bounds(self, bounds):
        with pytest.raises(ValueError):
            RandomGenerator(bounds)


@pytest.mark.skipif(not has_scipy, reason="Requires scipy to test PerturbationGenerator")
class TestPerturbation:

    @pytest.mark.parametrize("x0, bounds, scale",
                             [([0, -5, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, 7e6],
                                                                [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -500, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, 7e6],
                                                                     [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 9540.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, 7e6],
                                                                    [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [1000, 950], [0, 7e6],
                                                                   [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, np.inf],
                                                                   [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 954.54, 6.23e6], [[0, 1], [-100, 100], [950, 1000], [0, np.inf],
                                                            [-10, 0]], [1, 25, 2, 1e1, 5])
                              ])
    def test_invalid_bounds(self, x0, bounds, scale):
        with pytest.raises(ValueError):
            PerturbationGenerator(x0, bounds, scale)

    def test_call(self):
        np.random.seed(6)

        x0 = [0.01, -5, 954.54, 6.23e6, -3.45]
        bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
        scale = [1, 25, 2, 1e6, 5]
        gen = PerturbationGenerator(x0, bounds, scale)

        for _ in range(50):
            call = gen.generate(None)
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    def test_call_mean(self):
        np.random.seed(7687)

        x0 = [0.02, -5, 954.54, 6.23e6, -3.45]
        bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
        scale = [0.01, 5, 10, 1e5, 1]
        gen = PerturbationGenerator(x0, bounds, scale)

        calls = []
        for _ in range(5000):
            calls.append(gen.generate(None))

        assert np.all(np.isclose(np.mean(calls, 0) / x0, 1, rtol=0.1))


class TestBest:

    @pytest.fixture()
    def manager(self, request):
        class Manager:
            def __init__(self, result):
                self.result = Result(result, 56, None, None)

        return request.param, Manager(request.param)

    @pytest.mark.parametrize("manager", [None, [0, 0]], indirect=["manager"])
    def test_call(self, manager):
        point, man = manager
        gen = IncumbentGenerator(((1, 2),) * 2)
        call = gen.generate(man)

        assert len(call) == 2

        if point is None:
            assert all([1 < i < 2 for i in call])
        else:
            assert call == point


class TestExploitExplore:
    history_normal = {1:
                          {1: {'f_call_overall': 9,
                               'f_call_opt': 9,
                               'fx': 14.719,
                               'i_best': 1,
                               'fx_best': 14.719,
                               'x': [3.96, 7.06]},
                           2: {'f_call_overall': 18,
                               'f_call_opt': 18,
                               'fx': 8.897,
                               'i_best': 2,
                               'fx_best': 8.897,
                               'x': [1.35, 2.67]},
                           3: {'f_call_overall': 27,
                               'f_call_opt': 27,
                               'fx': 8.897,
                               'i_best': 2,
                               'fx_best': 8.897,
                               'x': [1.35, 9.78]},
                           4: {'f_call_overall': 36,
                               'f_call_opt': 36,
                               'fx': 8.897,
                               'i_best': 2,
                               'fx_best': 8.897,
                               'x': [1.35, 5.45]},
                           5: {'f_call_overall': 45,
                               'f_call_opt': 45,
                               'fx': 2.595,
                               'i_best': 5,
                               'fx_best': 2.595,
                               'x': [5.60, 2.48]},
                           },
                      2:
                          {1: {'f_call_overall': 9,
                               'f_call_opt': 9,
                               'fx': 5.911,
                               'i_best': 1,
                               'fx_best': 5.911,
                               'x': [1.65, 6.23]},
                           2: {'f_call_overall': 18,
                               'f_call_opt': 18,
                               'fx': 5.911,
                               'i_best': 1,
                               'fx_best': 5.911,
                               'x': [1.65, 6.23]},
                           3: {'f_call_overall': 27,
                               'f_call_opt': 27,
                               'fx': -8.03,
                               'i_best': 3,
                               'fx_best': -8.03,
                               'x': [1.91, 4.91]},
                           4: {'f_call_overall': 36,
                               'f_call_opt': 36,
                               'fx': -8.031,
                               'i_best': 3,
                               'fx_best': -8.031,
                               'x': [1.91, 4.91]},
                           5: {'f_call_overall': 45,
                               'f_call_opt': 45,
                               'fx': -8.031,
                               'i_best': 3,
                               'fx_best': -8.031,
                               'x': [1.91, 4.91]},
                           }
                      }

    @pytest.fixture()
    def manager(self, request):
        class OptLog:
            def __init__(self, history):
                self.history = history

            def get_history(self, opt):
                return self.history[opt]

        class Manager:
            def __init__(self, f_count, history):
                self.o_counter = len(history)
                self.f_counter = f_count
                self.opt_log = OptLog(history)

        return Manager(*request.param)

    # Test if only one opt

    @pytest.mark.parametrize("bounds, max_calls, focus, err", [(((-50, 4),) * 2, 5000, 0, ValueError),
                                                               (((-50, 4),) * 2, 5000, -1.3, ValueError),
                                                               (((-50, 4),) * 2, 1, 1, ValueError),
                                                               (((-50, 4),) * 2, 2.3, 1, TypeError),
                                                               (((50, 4),) * 2, 5000, 1, ValueError),
                                                               (((-50, 4),) * 2, 0, 1, ValueError)])
    def test_init(self, bounds, max_calls, focus, err):
        with pytest.raises(err):
            ExploitExploreGenerator(bounds, max_calls, focus)

    @pytest.mark.parametrize("manager", [(0, history_normal)], indirect=["manager"])
    def test_generate_random(self, manager):
        gen = ExploitExploreGenerator(((-1, 0),) * 2, 100, 1)
        call = gen.generate(manager)

        assert all([-1 < i < 0 for i in call])

    @pytest.mark.parametrize("manager", [(100, history_normal)], indirect=["manager"])
    def test_generate_incumbent(self, manager):
        gen = ExploitExploreGenerator(((0, 10),) * 2, 100, 1)
        call = gen.generate(manager)

        assert list(call) == [1.91, 4.91]

    @pytest.mark.parametrize("focus, manager, alpha", [(1.0, (50, history_normal), 0.50),
                                                       (2.0, (50, history_normal), 0.25),
                                                       (0.5, (25, history_normal), 0.50)],
                             indirect=["manager"])
    def test_generate_midpoint(self, focus, manager, alpha):
        gen = ExploitExploreGenerator(((0, 1e-10),) * 2, 100, focus)
        call = gen.generate(manager)

        b = np.array([1.91, 4.91])
        c = call

        assert np.isclose(np.cross(b, c), 0)
        assert np.all(np.isclose(c / b, alpha))

    @pytest.mark.parametrize("focus, manager, alpha", [(1.0, (50, {1: history_normal[1]}), 0.50),
                                                       (1.0, (50, {1: {}}), 0.50)],
                             indirect=["manager"])
    def test_generate_insuff_info(self, focus, manager, alpha):
        gen = ExploitExploreGenerator(((0, 1e-10),) * 2, 100, focus)
        call = gen.generate(manager)

        assert np.all(np.isclose(call, 0))


class TestSingle:

    @pytest.mark.parametrize("x, out", [(None, [1, 0]),
                                        ([2, 3], [2, 3])])
    def test_generate(self, x, out):
        gen = SinglePointGenerator(((1, 1 + 1e-10), (0, 1e-10)), x)
        call = gen.generate(None)

        assert np.all(np.isclose(call, out))
