

from typing import *
import pytest
import shutil
import warnings

from glompo.core.manager import GloMPOManager
from glompo.optimizers.baseoptimizer import BaseOptimizer
from glompo.optimizers.cmawrapper import CMAOptimizer
from glompo.generators.random import RandomGenerator
from glompo.convergence.nkillsafterconv import KillsAfterConvergence
from glompo.hunters.min_vic_trainpts import MinVictimTrainingPoints


class OptimizerTest1(BaseOptimizer):
    needscaler = False
    signal_pipe = None
    results_queue = None

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        pass


class OptimizerTest2:
    pass


class TestManger:

    @pytest.mark.parametrize("kwargs", [{'task': None},
                                        {'optimizers': {'default': OptimizerTest2}},
                                        {'optimizers': {'default': OptimizerTest1()}},
                                        {'optimizers': {'default': (OptimizerTest2, None, None)}},
                                        {'max_jobs': '2'},
                                        {'bounds': (0, 1)},
                                        {'x0_generator': OptimizerTest2()},
                                        {'x0_generator': OptimizerTest2},
                                        {'x0_generator': RandomGenerator},
                                        {'convergence_checker': KillsAfterConvergence},
                                        {'convergence_checker': OptimizerTest1()},
                                        {'killing_conditions': MinVictimTrainingPoints},
                                        {'killing_conditions': OptimizerTest1()},
                                        {'killing_conditions': OptimizerTest1},
                                        {'task_args': 564},
                                        {'task_kwargs': 66}
                                        ])
    def test_init_typeerr(self, kwargs):
        with pytest.raises(TypeError):
            keys = {**{'task': lambda x, y: x + y,
                       'optimizers': {'default': OptimizerTest1},
                       'n_parms': 2,
                       'bounds': ((0, 1), (0, 1)),
                       'overwrite_existing': True},
                    **kwargs}
            GloMPOManager(**keys)

    @pytest.mark.parametrize("kwargs", [{'optimizers': {'other': OptimizerTest1}},
                                        {'optimizers': {'default': (OptimizerTest1, None)}},
                                        {'optimizers': {'default': (OptimizerTest1, 65, 96)}},
                                        {'verbose': 6.7},
                                        {'n_parms': 6.0},
                                        {'n_parms': -1},
                                        {'n_parms': 0},
                                        {'max_jobs': -1},
                                        {'bounds': ((0, 1),)},
                                        {'bounds': ((1, 1), (0, 1))},
                                        {'bounds': ((2, 1), (0, 1))}])
    def test_init_valerr(self, kwargs):
        with pytest.raises(ValueError):
            keys = {**{'task': lambda x, y: x + y,
                       'optimizers': {'default': OptimizerTest1},
                       'n_parms': 2,
                       'bounds': ((0, 1), (0, 1)),
                       'overwrite_existing': True},
                    **kwargs}
            GloMPOManager(**keys)

    @pytest.mark.parametrize("kwargs", [{},
                                        {'optimizers': {'default': (OptimizerTest1, None, None)}},
                                        {'x0_generator': RandomGenerator(((0, 1), (0, 1)))},
                                        {'convergence_checker': KillsAfterConvergence()},
                                        {'max_jobs': 3},
                                        {'killing_conditions': MinVictimTrainingPoints(10)}])
    def test_init(self, kwargs):
        kwargs = {**{'task': lambda x, y: x + y,
                     'optimizers': {'default': OptimizerTest1},
                     'n_parms': 2,
                     'bounds': ((0, 1), (0, 1)),
                     'overwrite_existing': True},
                  **kwargs}
        GloMPOManager(**kwargs)

    @pytest.mark.parametrize("history_logging", [0, -5, 10, 23.56, 2.3])
    def test_init_clipping(self, history_logging):
        import numpy as np
        opt = GloMPOManager(lambda x, y: x + y,
                            2,
                            {'default': OptimizerTest1},
                            ((0, 1), (0, 1)),
                            overwrite_existing=True,
                            history_logging=history_logging)
        assert opt.history_logging == int(np.clip(int(history_logging), 0, 3))

    def test_init_optimizer_setup(self):
        manager = GloMPOManager(lambda x, y: x + y,
                                2,
                                {'default': OptimizerTest1,
                                 'early': (OptimizerTest1, None, None),
                                 'late': (OptimizerTest1, {'kwarg': 9}, None),
                                 'noisy': (OptimizerTest1, None, {'kwarg': 1923})},
                                ((0, 1), (0, 1)),
                                overwrite_existing=True)
        for opt in manager.optimizers.values():
            assert isinstance(opt, tuple)
            assert issubclass(opt[0], BaseOptimizer)
            assert isinstance(opt[1], dict)
            assert isinstance(opt[2], dict)

    def test_init_workingdir(self):
        with pytest.warns(UserWarning, match="Cannot parse working_dir"):
            GloMPOManager(lambda x, y: x + y,
                          2,
                          {'default': OptimizerTest1},
                          ((0, 1), (0, 1)),
                          working_dir=5,
                          overwrite_existing=True)

    def test_task_wrapper(self):
        def task(x, *args, calculate=False):
            if calculate:
                return sum([*args]) ** x

        wrapped_task = GloMPOManager._task_args_wrapper(task, (1, 2, 3, 4), {'calculate': True})

        for i in range(5):
            assert wrapped_task(i) == 10 ** i

    def test_redirect(self):
        import multiprocessing as mp

        def func():
            print("redirect_test")
            raise RuntimeError("redirect_test_error")

        wrapped_func = GloMPOManager._redirect(1, func)
        p = mp.Process(target=wrapped_func)
        p.start()
        p.join()

        with open("glompo_optimizer_printstreams/1_printstream.out", "r") as file:
            assert file.readline() == "redirect_test\n"

        with open("glompo_optimizer_printstreams/1_printstream.err", "r") as file:
            assert any(["redirect_test_error" in line for line in file.readlines()])


    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree("glompo_optimizer_printstreams")
        except FileNotFoundError:
            pass
