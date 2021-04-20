from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

import pytest
from glompo.core.optimizerlogger import BaseLogger
from glompo.opt_selectors.baseselector import BaseSelector
from glompo.opt_selectors.chain import ChainSelector
from glompo.opt_selectors.cycle import CycleSelector
from glompo.opt_selectors.random import RandomSelector
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult


class BasicOptimizer(BaseOptimizer):
    def __init__(self, a: int = 0, b: Optional[Sequence[float]] = None, c: Dict[str, Any] = None):
        self.a = a
        self.b = b
        self.c = c

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        pass

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def checkpoint_save(self, *args):
        pass

    def checkpoint_load(self, path: Union[Path, str]):
        pass


class OptimizerA(BasicOptimizer):
    pass


class OptimizerB(BasicOptimizer):
    pass


class BasicSelector(BaseSelector):

    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: BaseLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None]:
        return self.avail_opts[0]


class SpawnStopper:
    def __call__(self, *args, **kwargs):
        return False


class TestSelectors:
    @pytest.mark.parametrize('avail_opts', [[BasicOptimizer],
                                            [(BaseOptimizer, None, None)],
                                            [(BaseOptimizer, {'a': 5, 'b': [1, 2, 3], 'c':
                                                {'x': 5, 'y': BaseOptimizer}}, None)],
                                            [(BaseOptimizer, None, {'extra': 642})],
                                            [(BaseOptimizer, {'a': 5, 'b': [1, 2, 3], 'c':
                                                {'x': 5, 'y': BaseOptimizer}}, {'extra': 642})],
                                            [BasicOptimizer, BasicOptimizer],
                                            [BasicOptimizer, (BasicOptimizer, {'a': 1}, {'b': 3})],
                                            [(BasicOptimizer, {'a': 1}, {'b': 3}), BasicOptimizer]])
    def test_init(self, avail_opts):
        selector = BasicSelector(avail_opts)
        ret = selector.select_optimizer(None, None, 1)

        assert len(ret) == 3
        assert isinstance(ret, tuple)
        assert issubclass(ret[0], BaseOptimizer)
        assert isinstance(ret[1], dict)
        assert isinstance(ret[2], dict)

    @pytest.mark.parametrize("avail_opts, test, result", [([OptimizerA], OptimizerA, True),
                                                          ([OptimizerB], OptimizerA, False),
                                                          ([OptimizerA, OptimizerB], OptimizerA, True),
                                                          ([OptimizerA, OptimizerB], OptimizerB, True)])
    def test_contains(self, avail_opts, test, result):
        selector = BasicSelector(avail_opts)
        assert (test in selector) == result


class TestCycleSelector:

    def test_selection(self):
        selector = CycleSelector([OptimizerA, OptimizerB])

        for i in range(5):
            selection = selector.select_optimizer(None, None, 1)
            assert selection[0] == [OptimizerA, OptimizerB][i % 2]

    def test_no_space(self):
        selector = CycleSelector([(OptimizerA, {'workers': 2}, None), OptimizerB])

        selection = selector.select_optimizer(None, None, 1)
        assert selection is None

        selection = selector.select_optimizer(None, None, 1)
        assert selection is None

        selection = selector.select_optimizer(None, None, 2)
        assert selection[0] is OptimizerA

        selection = selector.select_optimizer(None, None, 1)
        assert selection[0] is OptimizerB

        selection = selector.select_optimizer(None, None, 1)
        assert selection is None

        selection = selector.select_optimizer(None, None, 2)
        assert selection[0] is OptimizerA

    def test_no_spawning(self):
        selector = CycleSelector([OptimizerA, OptimizerB], allow_spawn=SpawnStopper())
        assert selector.select_optimizer(None, None, 1) is False


class TestChain:

    def test_selection(self):
        class Manager:
            f_counter = 0

        selector = ChainSelector([OptimizerA,
                                  (OptimizerB, {'workers': 2}, None),
                                  OptimizerA,
                                  OptimizerB], [10, 20, 30])
        manager = Manager()

        selection = selector.select_optimizer(manager, None, 1)
        assert selection[0] == OptimizerA

        manager.f_counter = 3
        selection = selector.select_optimizer(manager, None, 1)
        assert selection[0] == OptimizerA

        manager.f_counter = 12
        selection = selector.select_optimizer(manager, None, 1)
        assert selection is None

        manager.f_counter = 15
        selection = selector.select_optimizer(manager, None, 2)
        assert selection[0] == OptimizerB

        manager.f_counter = 29
        selection = selector.select_optimizer(manager, None, 1)
        assert selection[0] == OptimizerA

        manager.f_counter = 30
        selection = selector.select_optimizer(manager, None, 1)
        assert selection[0] == OptimizerB

        manager.f_counter = 300
        selection = selector.select_optimizer(manager, None, 10)
        assert selection[0] == OptimizerB

    def test_no_spawning(self):
        selector = ChainSelector([OptimizerA, OptimizerB], [10], allow_spawn=SpawnStopper())
        assert selector.select_optimizer(None, None, 1) is False


class TestRandom:

    @pytest.mark.parametrize("workers_a", [1, 2])
    def test_no_space(self, workers_a):
        selector = RandomSelector([(OptimizerA, {'workers': workers_a}, None), (OptimizerB, {'workers': 2}, None)])

        if workers_a == 1:
            for i in range(10):
                assert selector.select_optimizer(None, None, 1)[0] == OptimizerA
        else:
            assert selector.select_optimizer(None, None, 1) is None

    def test_no_spawning(self):
        selector = RandomSelector([OptimizerA, OptimizerB], allow_spawn=SpawnStopper())
        assert selector.select_optimizer(None, None, 1) is False

    def test_selection(self):
        selector = RandomSelector([OptimizerA, OptimizerB])

        selected = set()
        for i in range(100):
            selected.add(selector.select_optimizer(None, None, 1)[0])

        assert selected == {OptimizerA, OptimizerB}
