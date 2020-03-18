from typing import Callable, Sequence, Tuple

import pytest

from glompo.core.manager import GloMPOManager
from glompo.optimizers.baseoptimizer import BaseOptimizer
from glompo.optimizers.cmawrapper import CMAOptimizer

from glompo.generators.random import RandomGenerator

from glompo.convergence.nkillsafterconv import KillsAfterConvergence


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
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5)

    def test_task2(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': OptimizerTest1},
                      ((0, 1), (0, 1)),
                      5)

    def test_optimizer1(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest2},
                          ((0, 1), (0, 1)),
                          5)

    def test_optimizer2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'other': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5)

    def test_optimizer3(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': (OptimizerTest1, None, None)},
                      ((0, 1), (0, 1)),
                      5)

    def test_optimizer4(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': (OptimizerTest2, None, None)},
                          ((0, 1), (0, 1)),
                          5)

    def test_optimizer5(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1,
                             'early': (OptimizerTest1, None, None),
                             'late': (OptimizerTest1, {'kwarg': 9}, None),
                             'noisy': (OptimizerTest1, None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['default'], tuple):
            errors.append("Not Tuple")
        if not opt.optimizers['default'][0] == OptimizerTest1:
            errors .append("First element not optimizer")
        if opt.optimizers['default'][1] != {}:
            errors.append("Second element not None")
        if opt.optimizers['default'][2] != {}:
            errors.append("Third element not None")
        assert not errors

    def test_optimizer6(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1,
                             'early': (OptimizerTest1, None, None),
                             'late': (OptimizerTest1, {'kwarg': 9}, None),
                             'noisy': (OptimizerTest1, None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['early'], tuple):
            errors.append("Not Tuple")
        if not opt.optimizers['early'][0] == OptimizerTest1:
            errors .append("First element not optimizer")
        if opt.optimizers['early'][1] is not None:
            errors.append("Second element not None")
        if opt.optimizers['early'][2] is not None:
            errors.append("Third element not None")
        assert not errors

    def test_optimizer7(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1,
                             'early': (OptimizerTest1, None, None),
                             'late': (OptimizerTest1, {'kwarg': 9}, None),
                             'noisy': (OptimizerTest1, None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['late'], tuple):
            errors.append("Not Tuple")
        if not opt.optimizers['late'][0] == OptimizerTest1:
            errors .append("First element not optimizer")
        if not isinstance(opt.optimizers['late'][1], dict):
            errors.append("Second element not dict")
        if opt.optimizers['late'][2] is not None:
            errors.append("Third element not None")
        assert not errors

    def test_optimizer8(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1,
                             'early': (OptimizerTest1, None, None),
                             'late': (OptimizerTest1, {'kwarg': 9}, None),
                             'noisy': (OptimizerTest1, None, {'kwarg': 1923})},
                            ((0, 1), (0, 1)),
                            5)
        errors = []
        if not isinstance(opt.optimizers['noisy'], tuple):
            errors.append("Not Tuple")
        if not opt.optimizers['noisy'][0] == OptimizerTest1:
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
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          0)

    def test_maxjobs_neg(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          -5)

    def test_maxjobs_float(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          2.5646)

    def test_maxjobs_nan(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          None)

    def test_bounds_diffdims(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x: x + 1,
                          1,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5)

    def test_bounds_equalminmax1(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((1, 1), (0, 1)),
                          5)

    def test_bounds_equalminmax2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (1, 1)),
                          5)

    def test_bounds_equalminmax3(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((1, 1), (1, 1)),
                          5)

    def test_bounds_wrongminmax1(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((2, 1), (0, 1)),
                          5)

    def test_bounds_wrongminmax2(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (2, 1)),
                          5)

    def test_bounds_wrongminmax3(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((2, 1), (2, 1)),
                          5)

    def test_x0crit1(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': OptimizerTest1},
                      ((0, 1), (0, 1)),
                      5,
                      x0_generator=RandomGenerator(((0, 1), (0, 1))))

    def test_x0crit2(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5,
                          x0_generator=OptimizerTest2())

    def test_x0crit3(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5,
                          x0_generator=OptimizerTest2)

    def test_x0crit4(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5,
                          x0_generator=RandomGenerator)

    def test_convcrit1(self):
        GloMPOManager(lambda x, y: x + y,
                      2,
                      {'default': OptimizerTest1},
                      ((0, 1), (0, 1)),
                      5,
                      convergence_checker=KillsAfterConvergence())

    def test_convcrit2(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5,
                          convergence_checker=OptimizerTest2())

    def test_convcrit3(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5,
                          convergence_checker=OptimizerTest2)

    def test_convcrit4(self):
        with pytest.raises(TypeError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5,
                          convergence_checker=KillsAfterConvergence)

    def test_history_logging1(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=0)
        assert opt.history_logging == 0

    def test_history_logging2(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=-5)
        assert opt.history_logging == 0

    def test_history_logging3(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=10)
        assert opt.history_logging == 3

    def test_history_logging4(self):
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1},
                            ((0, 1), (0, 1)),
                            5,
                            history_logging=23.56)
        assert opt.history_logging == 3

    def test_history_logging5(self):
        with pytest.raises(ValueError):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          5,
                          history_logging='x')


class TestManagerMethods:
    optimizer = GloMPOManager(task=lambda x, y: x ** 2 + y ** 2,
                              n_parms=2,
                              optimizers={'default': CMAOptimizer},
                              bounds=((-5, 5), (-3, 3)),
                              max_jobs=2,
                              overwrite_existing=True)
