

from typing import *
from time import sleep
import pytest
import shutil
import sys

from glompo.core.manager import GloMPOManager
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from glompo.generators.random import RandomGenerator
from glompo.convergence import KillsAfterConvergence, MaxSeconds, MaxFuncCalls
from glompo.hunters import MinVictimTrainingPoints, GPRSuitable, ValBelowGPR
from glompo.common.namedtuples import *
from glompo.generators.basegenerator import BaseGenerator

import numpy as np
import yaml


class OptimizerTest1(BaseOptimizer):
    needscaler = False
    signal_pipe = None
    results_queue = None

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        pass

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def save_state(self, *args):
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

    @pytest.mark.mini
    def test_mwe(self):
        class SteepestGradient(BaseOptimizer):

            needscaler = False

            def __init__(self, max_iters, gamma, precision,
                         opt_id=None, signal_pipe=None, results_queue=None,
                         pause_flag=None):
                super().__init__(opt_id, signal_pipe, results_queue, pause_flag)
                self.max_iters = max_iters
                self.gamma = np.array(gamma)
                self.precision = precision
                self.terminate = False
                self.reason = None
                self.current_x = None

            def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                         callbacks: Callable = None, **kwargs) -> MinimizeResult:

                next_x = np.array(x0)
                it = 0
                while not self.terminate:
                    if self._pause_signal:
                        self._pause_signal.wait()
                    it += 1
                    if self._signal_pipe:
                        self.check_messages()
                    self.current_x = next_x
                    fx, dx, dy = function(self.current_x)
                    if self._results_queue:
                        self.push_iter_result(IterationResult(self._opt_id,
                                                              it,
                                                              1,
                                                              self.current_x,
                                                              fx,
                                                              False if it <= self.max_iters else True))
                    print(f"Iter {it}: x = {self.current_x}, fx = {fx}")
                    grad = np.array([dx, dy])
                    next_x = self.current_x - self.gamma * grad

                    step = next_x - self.current_x
                    if np.all(np.abs(step) <= self.precision):
                        self.reason = "xtol condition"
                        break
                    if it >= self.max_iters:
                        self.reason = "imax condition"
                        break
                if self._signal_pipe:
                    self.message_manager(0)
                print(f"Stopping due to {self.reason}")
                return next_x, self.reason

            def callstop(self, *args):
                self.terminate = True
                self.reason = "manager termination"

            def save_state(self, *args):
                with open("steepgrad_savestate.yml" "w+") as file:
                    data = {"Settings": {"max_iters": self.max_iters,
                                         "gamma": self.gamma,
                                         "precision": self.precision},
                            "Last_x": self.current_x}
                    yaml.dump(data, file, default_flow_style=False)

            def push_iter_result(self, itres):
                self._results_queue.put(itres)

        # Task
        def f(pt, delay=0):
            x, y = pt
            calc = -np.cos(0.2 * x)
            calc *= np.exp(-x ** 2 / 5000)
            calc /= 50 * np.sqrt(2 * np.pi)
            calc += 1e-6 * y ** 2
            sleep(delay)
            return calc, df_dx(pt), df_dy(pt)

        def df_dx(pt):
            x, y = pt
            calc = np.exp(-x ** 2 / 5000)
            calc *= x * np.cos(0.2 * x)
            calc /= 125000 * np.sqrt(2 * np.pi)
            calc += 0.00159577 * np.exp(-x ** 2 / 5000) * np.sin(0.2 * x)
            return calc

        def df_dy(pt):
            x, y = pt
            calc = 2e-6 * y
            return calc

        # x0_generator
        class IntervalGenerator(BaseGenerator):
            def __init__(self):
                self.count = -1

            def generate(self, *args, **kwargs) -> np.ndarray:
                self.count += 1
                x = np.random.choice([-1, 1]) * 30 * self.count + np.random.uniform(-2, 2)
                y = np.random.uniform(-100, 100)
                return x, y

        manager = GloMPOManager(task=f,
                                n_parms=2,
                                optimizers={'default': (SteepestGradient, {'max_iters': 10000,
                                                                           'precision': 1e-8,
                                                                           'gamma': [100, 100000]}, None)},
                                bounds=((-100, 100), (-100, 100)),
                                working_dir='tests/outputs',
                                overwrite_existing=True,
                                max_jobs=3,
                                task_kwargs={'delay': 0.1},
                                convergence_checker=KillsAfterConvergence(2, 1) + MaxFuncCalls(10000) + MaxSeconds(
                                    5 * 60),
                                x0_generator=IntervalGenerator(),
                                killing_conditions=GPRSuitable(0.1) * MinVictimTrainingPoints(10) * ValBelowGPR(),
                                history_logging=3,
                                visualisation=False,
                                visualisation_args=None,
                                verbose=0)
        result = manager.start_manager()
        assert np.all(np.isclose(result.x, 0, atol=1e-6))
        assert np.isclose(result.fx, -0.00797884560802864)
        assert result.origin['opt_id'] == 1
        assert result.origin['type'] == 'SteepestGradient'

    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree("glompo_optimizer_printstreams", ignore_errors=True)
            is_save = '--save-outs' in sys.argv
            is_mini = '--run-minimize' in sys.argv
            if is_mini and not is_save:
                shutil.rmtree("tests/outputs", ignore_errors=True)
        except FileNotFoundError:
            pass
