import numpy as np
import pytest

from glompo.common.namedtuples import Bound, Result
from glompo.generators.basinhopping import BasinHoppingGenerator
from glompo.generators.best import IncumbentGenerator
from glompo.generators.exploit_explore import ExploitExploreGenerator
from glompo.generators.random import RandomGenerator
from glompo.generators.single import SinglePointGenerator

try:
    from glompo.generators.peterbation import PerturbationGenerator
    from glompo.generators.annealing import AnnealingGenerator
    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False


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


@pytest.mark.skipif(not HAS_SCIPY, reason="Requires scipy to test PerturbationGenerator")
class TestPerturbation:

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
    bests = {0: {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0},
             1: {'opt_id': 1, 'x': [5.60, 2.48], 'fx': 2.595, 'type': '', 'call_id': 10},
             2: {'opt_id': 2, 'x': [1.91, 4.91], 'fx': -8.03, 'type': '', 'call_id': 6}}

    @pytest.fixture()
    def manager(self, request):
        class OptLog:
            def __init__(self, bests):
                self._best_iters = bests

            def get_best_iter(self, opt_id):
                return self._best_iters[opt_id]

        class Manager:
            def __init__(self, f_count, bests):
                self.o_counter = len(bests) - 1
                self.f_counter = f_count
                self.opt_log = OptLog(bests)

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

    @pytest.mark.parametrize("manager", [(0, bests)], indirect=["manager"])
    def test_generate_random(self, manager):
        gen = ExploitExploreGenerator(((-1, 0),) * 2, 100, 1)
        call = gen.generate(manager)

        assert all([-1 < i < 0 for i in call])

    @pytest.mark.parametrize("manager", [(100, bests)], indirect=["manager"])
    def test_generate_incumbent(self, manager):
        gen = ExploitExploreGenerator(((0, 10),) * 2, 100, 1)
        call = gen.generate(manager)

        assert list(call) == [1.91, 4.91]

    @pytest.mark.parametrize("focus, manager, alpha", [(1.0, (50, bests), 0.50),
                                                       (2.0, (50, bests), 0.25),
                                                       (0.5, (25, bests), 0.50)],
                             indirect=["manager"])
    def test_generate_midpoint(self, focus, manager, alpha):
        gen = ExploitExploreGenerator(((0, 1e-10),) * 2, 100, focus)
        call = gen.generate(manager)

        b = np.array([1.91, 4.91])
        c = call

        assert np.isclose(np.cross(b, c), 0)
        assert np.all(np.isclose(c / b, alpha))

    @pytest.mark.parametrize("manager", [(50, {0: bests[0], 1: bests[1]})], indirect=["manager"])
    def test_generate_insuff_info(self, manager):
        gen = ExploitExploreGenerator(((0, 1e-10),) * 2, 100, 1.0)
        call = gen.generate(manager)

        assert np.all(np.isclose(call, 0))


class TestSingle:

    @pytest.mark.parametrize("x, out", [(None, [1, 0]),
                                        ([2, 3], [2, 3])])
    def test_generate(self, x, out):
        gen = SinglePointGenerator(((1, 1 + 1e-10), (0, 1e-10)), x)
        call = gen.generate(None)

        assert np.all(np.isclose(call, out))


class TestBasinHopping:

    @pytest.fixture(scope='function')
    def manager(self, request):
        n_parms = 3

        class OptLog:
            def __init__(self):
                self._best_iters = {0: {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0}}
                for i in range(1, request.param + 1):
                    self._best_iters[i] = {'opt_id': i,
                                           'x': np.array([1 / i] * n_parms),
                                           'fx': 100 / i,
                                           'type': 'FakeOptimizer',
                                           'call_id': 30 * i}

                self._best_iter = self._best_iters[request.param]

            def get_best_iter(self):
                return self._best_iter

        class Manager:
            def __init__(self):
                self.n_parms = n_parms
                self.bounds = [(0, 1)] * self.n_parms
                self.o_counter = request.param + 1
                self.opt_log = OptLog()

        return Manager()

    @pytest.fixture(scope='function')
    def generator(self):
        return BasinHoppingGenerator(interval=4)

    @pytest.mark.parametrize("manager", [0], indirect=["manager"])
    def test_random_generate(self, manager, generator, caplog):
        caplog.set_level('DEBUG', logger='glompo.generator')
        generator.generate(manager)
        assert "Iteration history not found, returning random start location." in caplog.messages

    @pytest.mark.parametrize("manager", [1], indirect=["manager"])
    def test_improve(self, manager, generator):
        generator.generate(manager)
        assert generator.state == {'opt_id': 1, 'fx': 100}

    @pytest.mark.parametrize("manager", [2], indirect=["manager"])
    def test_monte_carlo_accept(self, manager, generator, monkeypatch):
        monkeypatch.setattr('numpy.random.random', lambda: 0)
        generator.step_size = 0  # Force return of location 1
        ret = generator.generate(manager)
        assert generator.state == {'opt_id': 2, 'fx': 50}
        assert np.all(ret == [1] * manager.n_parms)

    @pytest.mark.parametrize("manager", [3], indirect=["manager"])
    def test_step_adjust(self, manager, generator):
        for i in range(3):
            generator.n_accept += 1
            generator.generate(manager)
            assert generator.step_size == [0.45, 0.45, 0.5][i]


class TestAnnealing:

    @pytest.fixture(scope='function')
    def manager(self):
        class OptLog:
            def __init__(self):
                self._best_iters = {0: {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0},
                                    1: {'opt_id': 1,
                                        'x': np.array([100] * 5),
                                        'fx': 500,
                                        'type': 'FakeOptimizer',
                                        'call_id': 9}}
                self._best_iter = self._best_iters[1]
                self._f_counter = 9

            def get_best_iter(self):
                return self._best_iter

        class Manager:
            def __init__(self):
                self.f_counter = 9
                self.opt_log = OptLog()

        return Manager()

    @pytest.fixture()
    def generator(self):
        return AnnealingGenerator([(0, 100)] * 5, lambda x: np.sum(x))

    def test_generate(self, manager, generator):
        generator.iter += 1
        generator.generate(manager)

        assert generator.state.current_energy < 500
        assert np.all(generator.state.current_location < 100)
        assert generator.temperature < 5230
        assert manager.f_counter == manager.opt_log._f_counter == 19

    def test_reset(self, generator):
        generator.reset_temperature()

        assert generator.iter == 0
        assert generator.temperature == 5230
