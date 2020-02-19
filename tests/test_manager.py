from typing import Callable, Sequence, Tuple

import pytest

from ..core.manager import GloMPOOptimizer
from ..optimizers.baseoptimizer import BaseOptimizer


class TestMangerInit:

    class OptimizerTest1(BaseOptimizer):
        needscaler = False
        signal_pipe = None
        results_queue = None

        def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                     callbacks: Callable = None, **kwargs):
            pass

    class OptimizerTest2:
        pass

    def test_task_raises(self):
        with pytest.raises(TypeError):
            GloMPOOptimizer(None,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5)

    def test_task_notraises(self):
        GloMPOOptimizer(lambda x, y: x+y,
                        2,
                        {'default': self.OptimizerTest1()},
                        ((0, 1), (0, 1)),
                        5)

    def test_optimizer_raisesnotbase(self):
        with pytest.raises(TypeError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest2()},
                            ((0, 1), (0, 1)),
                            5)

    def test_optimizer_raisesnotdefault(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'other': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5)

    def test_maxjobs_0(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            0)

    def test_maxjobs_neg(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            -5)

    def test_maxjobs_float(self):
        with pytest.raises(TypeError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            2.5646)

    def test_maxjobs_nan(self):
        with pytest.raises(TypeError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            None)

    def test_bounds_diffdims(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x: x+1,
                            1,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5)

    def test_bounds_equalminmax1(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((1, 1), (0, 1)),
                            5)

    def test_bounds_equalminmax2(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (1, 1)),
                            5)

    def test_bounds_equalminmax3(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((1, 1), (1, 1)),
                            5)

    def test_bounds_wrongminmax1(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((2, 1), (0, 1)),
                            5)

    def test_bounds_wrongminmax2(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (2, 1)),
                            5)

    def test_bounds_wrongminmax3(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((2, 1), (2, 1)),
                            5)

    def test_x0crit1(self):
        GloMPOOptimizer(lambda x, y: x+y,
                        2,
                        {'default': self.OptimizerTest1()},
                        ((0, 1), (0, 1)),
                        5,
                        x0_criteria='rand')

    def test_x0crit2(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            x0_criteria='not_allowed')

    def test_convcrit1(self):
        GloMPOOptimizer(lambda x, y: x+y,
                        2,
                        {'default': self.OptimizerTest1()},
                        ((0, 1), (0, 1)),
                        5,
                        convergence_criteria='sing_conv')

    def test_convcrit2(self):
        with pytest.raises(ValueError):
            GloMPOOptimizer(lambda x, y: x+y,
                            2,
                            {'default': self.OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            convergence_criteria='not_allowed')
