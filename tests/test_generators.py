import numpy as np
import pytest

from glompo.common.namedtuples import Result
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


@pytest.fixture(scope='function')
def manager(request):
    if not hasattr(request, 'param'):
        request.param = None

    class FakeManager:
        def __init__(self, result=None):
            self.bounds = np.array([(-10, -5)] * 42)
            self.n_parms = 42
            self.result = Result(result, 56, None, None)

    return FakeManager(request.param)


@pytest.mark.parametrize('generator, gen_args', [pytest.param(RandomGenerator, []),
                                                 pytest.param(PerturbationGenerator, [np.linspace(-10, -5, 42),
                                                                                      np.linspace(0.01, 5, 42)],
                                                              marks=pytest.mark.skipif(not HAS_SCIPY,
                                                                                       reason="Requires scipy")),
                                                 pytest.param(IncumbentGenerator, []),
                                                 pytest.param(SinglePointGenerator, [])])
def test_call(generator, gen_args, manager):
    np.random.seed(1)

    gen = generator(*gen_args)
    x0 = gen.generate(manager)

    assert x0.size == manager.bounds.shape[0]
    assert np.all([(x0 > manager.bounds[:, 0]) & (x0 < manager.bounds[:, 1])])


@pytest.mark.parametrize("manager", [np.array([-7.2] * 42)], indirect=["manager"])
def test_incumbent(manager):
    gen = IncumbentGenerator()
    call = gen.generate(manager)
    assert np.all(call == manager.result.x)


def test_single(manager):
    gen = SinglePointGenerator()
    call_a = gen.generate(manager)
    call_b = gen.generate(manager)
    assert np.all(call_a == call_b)


class TestExploitExplore:
    bests = {0: {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0},
             1: {'opt_id': 1, 'x': [5.60, 2.48], 'fx': 2.595, 'type': '', 'call_id': 10},
             2: {'opt_id': 2, 'x': [1.91, 4.91], 'fx': -8.03, 'type': '', 'call_id': 6}}

    @pytest.fixture()
    def manager_with_log(self, request):
        class OptLog:
            def __init__(self, bests):
                self.best_iters = bests

            def get_best_iter(self, opt_id):
                return self.best_iters[opt_id]

        class Manager:
            def __init__(self, f_count, bests):
                self.o_counter = len(bests) - 1
                self.f_counter = f_count
                self.opt_log = OptLog(bests)
                self.n_parms = 2
                self.bounds = np.array([(0, 10)] * 2)

        return Manager(*request.param)

    @pytest.mark.parametrize("manager_with_log", [(0, bests)], indirect=["manager_with_log"])
    def test_generate_random(self, manager_with_log):
        gen = ExploitExploreGenerator(100, 1)
        call = gen.generate(manager_with_log)
        assert np.all([(call > 0) & (call < 10)])

    @pytest.mark.parametrize("manager_with_log", [(100, bests)], indirect=["manager_with_log"])
    def test_generate_incumbent(self, manager_with_log):
        gen = ExploitExploreGenerator(100, 1)
        call = gen.generate(manager_with_log)

        assert np.all(call == [1.91, 4.91])

    @pytest.mark.parametrize("focus, manager_with_log, alpha", [(1.0, (50, bests), 0.50),
                                                                (2.0, (50, bests), 0.25),
                                                                (0.5, (25, bests), 0.50)],
                             indirect=["manager_with_log"])
    def test_generate_midpoint(self, focus, manager_with_log, alpha):
        gen = ExploitExploreGenerator(100, focus)
        manager_with_log.bounds = ((0, 1e-10),) * 2
        call = gen.generate(manager_with_log)

        b = np.array([1.91, 4.91])
        c = call

        assert np.isclose(np.cross(b, c), 0)
        assert np.all(np.isclose(c / b, alpha))

    @pytest.mark.parametrize("manager_with_log", [(50, {0: bests[0], 1: bests[1]})], indirect=["manager_with_log"])
    def test_generate_insuff_info(self, manager_with_log):
        gen = ExploitExploreGenerator(100, 1.0)
        manager_with_log.bounds = ((0, 1e-10),) * 2
        call = gen.generate(manager_with_log)

        assert np.all(np.isclose(call, 0))


class TestBasinHopping:

    @pytest.fixture(scope='function')
    def manager(self, request):
        n_parms = 3

        class OptLog:
            def __init__(self):
                self.best_iters = {0: {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0}}
                for i in range(1, request.param + 1):
                    self.best_iters[i] = {'opt_id': i,
                                          'x': np.array([1 / i] * n_parms),
                                          'fx': 100 / i,
                                          'type': 'FakeOptimizer',
                                          'call_id': 30 * i}

                self._best_iter = self.best_iters[request.param]

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
                self.best_iters = {0: {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0},
                                   1: {'opt_id': 1,
                                       'x': np.array([100] * 5),
                                       'fx': 500,
                                       'type': 'FakeOptimizer',
                                       'call_id': 9}}
                self._best_iter = self.best_iters[1]
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
