from typing import Callable, Sequence, Tuple

import pytest

from glompo.core.manager import GloMPOManager
from glompo.optimizers.baseoptimizer import BaseOptimizer
from glompo.optimizers.cmawrapper import CMAOptimizer


class OptimizerTest1(BaseOptimizer):
    needscaler = False
    signal_pipe = None
    results_queue = None

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        pass


class OptimizerTest2:
    pass


class TestMangerInit:
    def test_task1(self):
        with pytest.raises(TypeError):
            GloMPOManager(None,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5)

    def test_task2(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': OptimizerTest1()},
                      ((0, 1), (0, 1)),
                      5)

    def test_optimizer1(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest2()},
                          ((0, 1), (0, 1)),
                          5)

    def test_optimizer2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'other': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5)

    def test_optimizer3(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': (OptimizerTest1(), None, None)},
                      ((0, 1), (0, 1)),
                      5)

    def test_optimizer4(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': (OptimizerTest2(), None, None)},
                          ((0, 1), (0, 1)),
                          5)

    def test_optimizer5(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1(),
                               'early': (OptimizerTest1(), None, None),
                               'late': (OptimizerTest1(), {'kwarg': 9}, None),
                               'noisy': (OptimizerTest1(), None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['default'], tuple):
            errors.append("Not Tuple")
        if not isinstance(opt.optimizers['default'][0], OptimizerTest1):
            errors .append("First element not optimizer")
        if opt.optimizers['default'][1] is not None:
            errors.append("Second element not None")
        if opt.optimizers['default'][2] is not None:
            errors.append("Third element not None")
        assert not errors

    def test_optimizer6(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1(),
                               'early': (OptimizerTest1(), None, None),
                               'late': (OptimizerTest1(), {'kwarg': 9}, None),
                               'noisy': (OptimizerTest1(), None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['early'], tuple):
            errors.append("Not Tuple")
        if not isinstance(opt.optimizers['early'][0], OptimizerTest1):
            errors .append("First element not optimizer")
        if opt.optimizers['early'][1] is not None:
            errors.append("Second element not None")
        if opt.optimizers['early'][2] is not None:
            errors.append("Third element not None")
        assert not errors

    def test_optimizer7(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1(),
                               'early': (OptimizerTest1(), None, None),
                               'late': (OptimizerTest1(), {'kwarg': 9}, None),
                               'noisy': (OptimizerTest1(), None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['late'], tuple):
            errors.append("Not Tuple")
        if not isinstance(opt.optimizers['late'][0], OptimizerTest1):
            errors .append("First element not optimizer")
        if not isinstance(opt.optimizers['late'][1], dict):
            errors.append("Second element not dict")
        if opt.optimizers['late'][2] is not None:
            errors.append("Third element not None")
        assert not errors

    def test_optimizer8(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1(),
                               'early': (OptimizerTest1(), None, None),
                               'late': (OptimizerTest1(), {'kwarg': 9}, None),
                               'noisy': (OptimizerTest1(), None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['noisy'], tuple):
            errors.append("Not Tuple")
        if not isinstance(opt.optimizers['noisy'][0], OptimizerTest1):
            errors .append("First element not optimizer")
        if opt.optimizers['noisy'][1] is not None:
            errors.append("Second element not None")
        if not isinstance(opt.optimizers['noisy'][2], dict):
            errors.append("Third element not dict")
        assert not errors

    def test_maxjobs_0(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          0)

    def test_maxjobs_neg(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          -5)

    def test_maxjobs_float(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          2.5646)

    def test_maxjobs_nan(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          None)

    def test_bounds_diffdims(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x: x + 1,
                          1,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5)

    def test_bounds_equalminmax1(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((1, 1), (0, 1)),
                          5)

    def test_bounds_equalminmax2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (1, 1)),
                          5)

    def test_bounds_equalminmax3(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((1, 1), (1, 1)),
                          5)

    def test_bounds_wrongminmax1(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((2, 1), (0, 1)),
                          5)

    def test_bounds_wrongminmax2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (2, 1)),
                          5)

    def test_bounds_wrongminmax3(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((2, 1), (2, 1)),
                          5)

    def test_x0crit1(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': OptimizerTest1()},
                      ((0, 1), (0, 1)),
                      5,
                      x0_criteria='rand')

    def test_x0crit2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5,
                          x0_criteria='not_allowed')

    def test_convcrit1(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': OptimizerTest1()},
                      ((0, 1), (0, 1)),
                      5,
                      convergence_criteria='sing_conv')

    def test_convcrit2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5,
                          convergence_criteria='not_allowed')

    def test_omax1(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5,
                          omax='x')

    def test_omax2(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            omax=4.3)
        assert opt.omax == 4

    def test_omax3(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            omax=-6)
        assert opt.omax == 1

    def test_omax4(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            omax=0)
        assert opt.omax == 1

    def test_fmax1(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5,
                          fmax='x')

    def test_fmax2(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            fmax=4.3)
        assert opt.fmax == 4

    def test_fmax3(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            fmax=-6)
        assert opt.fmax == 1

    def test_fmax4(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            fmax=0)
        assert opt.fmax == 1

    def test_tmax1(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5,
                          tmax='x')

    def test_tmax2(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            tmax=4.3)
        assert opt.tmax == 4

    def test_tmax3(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            tmax=-6)
        assert opt.tmax == 1

    def test_tmax4(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            tmax=0)
        assert opt.tmax == 1

    def test_history_logging1(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=0)
        assert opt.history_logging == 0

    def test_history_logging2(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=-5)
        assert opt.history_logging == 0

    def test_history_logging3(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=10)
        assert opt.history_logging == 3

    def test_history_logging4(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1()},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=23.56)
        assert opt.history_logging == 3

    def test_history_logging5(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1()},
                          ((0, 1), (0, 1)),
                          5,
                          history_logging='x')


class TestManagerMethods:
    optimizer = GloMPOManager(task=lambda x, y: x ** 2 + y ** 2,
                              n_parms=2,
                              optimizers={'default': CMAOptimizer(0)},
                              bounds=((-5, 5), (-3, 3)),
                              max_jobs=2)

    def test_x0rand(self):
        self.optimizer.x0_criteria = 'rand'
        x0 = self.optimizer._generate_x0()
        for i, x in enumerate(x0):
            assert x > self.optimizer.bounds[i].min or x < self.optimizer.bounds[i].max
