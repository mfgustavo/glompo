
from typing import *

import pytest

from glompo.core.optimizerlogger import OptimizerLogger
from glompo.opt_selectors.baseselector import BaseSelector
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult


class BasicOptimizer(BaseOptimizer):

    needscaler = False

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

    def save_state(self, *args):
        pass


class BasicSelector(BaseSelector):

    def select_optimizer(self, manager: 'GloMPOManager', log: OptimizerLogger) -> Tuple[Type[BaseOptimizer],
                                                                                        Dict[str, Any], Dict[str, Any]]:
        return self.avail_opts[0]


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
        ret = selector.select_optimizer(None, None)

        assert len(ret) == 3
        assert isinstance(ret, tuple)
        assert issubclass(ret[0], BaseOptimizer)
        assert isinstance(ret[1], dict)
        assert isinstance(ret[2], dict)
