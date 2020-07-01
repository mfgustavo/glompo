

from typing import *
from time import sleep
import shutil
import sys
import os
import multiprocessing as mp

import numpy as np
import yaml
import pytest

from glompo.core.manager import GloMPOManager
from glompo.core.optimizerlogger import OptimizerLogger
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from glompo.generators import RandomGenerator, BaseGenerator
from glompo.convergence import BaseChecker, KillsAfterConvergence, MaxOptsStarted, MaxFuncCalls, MaxSeconds
from glompo.hunters import BaseHunter, MinIterations
from glompo.common.namedtuples import *
from glompo.common.wrappers import process_print_redirect, task_args_wrapper
from glompo.opt_selectors import BaseSelector, CycleSelector


class DummySelector(BaseSelector):
    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: OptimizerLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None]:
        pass


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


class MessagingOptimizer(BaseOptimizer):

    needscaler = False

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        self.message_manager(9, "This is a test of the GloMPO signalling system")
        self.message_manager(0)
        sleep(1)

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def save_state(self, *args):
        pass


class SilentOptimizer(BaseOptimizer):

    needscaler = False

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        sleep(1)

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def save_state(self, *args):
        pass


class HangingOptimizer(BaseOptimizer):

    needscaler = False

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        while True:
            pass

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def save_state(self, *args):
        pass


class HangOnEndOptimizer(BaseOptimizer):

    needscaler = False

    def __init__(self, opt_id: int = None, signal_pipe=None, results_queue=None, pause_flag=None, workers=1):
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers)
        self.constant = opt_id

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        i = 0
        while True:
            i += 1
            self.check_messages()
            sleep(0.1)
            self.push_iter_result(IterationResult(self._opt_id, i, 1, [i], np.sin(i/(2*np.pi))+self.constant, False))

    def push_iter_result(self, ir):
        self._results_queue.put(ir)

    def callstop(self, *args):
        print("Hanging Callstop Activated")
        while True:
            pass

    def save_state(self, *args):
        pass


class ErrorOptimizer(BaseOptimizer):

    needscaler = False

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        raise RuntimeError("This is a test of the GloMPO error handling service")

    def push_iter_result(self, ir):
        pass

    def callstop(self, *args):
        pass

    def save_state(self, *args):
        pass


class TrueHunter(BaseHunter):
    def __init__(self, target: int):
        super().__init__()
        self.target = target

    def __call__(self, log, hunter_opt_id, victim_opt_id) -> bool:
        return victim_opt_id == self.target


class ErrorChecker(BaseChecker):
    def __call__(self, manager: 'GloMPOManager') -> bool:
        raise RuntimeError("This is a test of the GloMPO error management system.")


@pytest.mark.parametrize('backend', ['processes', 'threads'])
class TestManager:

    base_wd = os.getcwd()

    def setup_method(self):
        os.chdir(self.base_wd)
        shutil.rmtree("tests/temp_outputs", ignore_errors=True)

    @pytest.mark.parametrize("kwargs", [{'task': None},
                                        {'optimizer_selector': {'default': OptimizerTest2}},
                                        {'optimizer_selector': OptimizerTest1()},
                                        {'max_jobs': '2'},
                                        {'bounds': (0, 1)},
                                        {'x0_generator': OptimizerTest2()},
                                        {'x0_generator': OptimizerTest2},
                                        {'x0_generator': RandomGenerator},
                                        {'convergence_checker': KillsAfterConvergence},
                                        {'convergence_checker': OptimizerTest1()},
                                        {'killing_conditions': MinIterations},
                                        {'killing_conditions': OptimizerTest1()},
                                        {'killing_conditions': OptimizerTest1},
                                        {'task_args': 564},
                                        {'task_kwargs': 66},
                                        {'gpr_training': 200}
                                        ])
    def test_init_typeerr(self, kwargs, backend):
        with pytest.raises(TypeError):
            keys = {**{'task': lambda x, y: x + y,
                       'optimizer_selector': DummySelector([OptimizerTest1]),
                       'bounds': ((0, 1), (0, 1)),
                       'overwrite_existing': True},
                    **kwargs}
            GloMPOManager(backend=backend, **keys)

    @pytest.mark.parametrize("kwargs", [{'bounds': ((0, 0), (0, 1))},
                                        {'bounds': ((1, 0), (0, 1))},
                                        {'max_jobs': -1}])
    def test_init_valerr(self, kwargs, backend):
        with pytest.raises(ValueError):
            keys = {**{'task': lambda x, y: x + y,
                       'optimizer_selector': DummySelector([OptimizerTest1]),
                       'bounds': ((0, 1), (0, 1)),
                       'overwrite_existing': True},
                    **kwargs}
            GloMPOManager(backend=backend, **keys)

    def test_invalid_backend(self, backend):
        with pytest.warns(UserWarning, match="Unable to parse backend"):
            keys = {'task': lambda x, y: x + y,
                    'optimizer_selector': DummySelector([OptimizerTest1]),
                    'bounds': ((0, 1), (0, 1)),
                    'overwrite_existing': True}
            GloMPOManager(backend='magic', **keys)

    def test_init_filexerr(self, backend):
        with open("glompo_manager_log.yml", "w+"):
            pass

        with pytest.raises(FileExistsError):
            GloMPOManager(task=lambda x, y: x + y,
                          optimizer_selector=DummySelector([OptimizerTest1]),
                          bounds=((0, 1), (0, 1)),
                          overwrite_existing=False,
                          backend=backend)

        os.remove("glompo_manager_log.yml")

    @pytest.mark.parametrize("kwargs", [{},
                                        {'x0_generator': RandomGenerator(((0, 1), (0, 1)))},
                                        {'convergence_checker': KillsAfterConvergence()},
                                        {'max_jobs': 3},
                                        {'killing_conditions': MinIterations(10)}])
    def test_init(self, kwargs, backend):
        kwargs = {**{'task': lambda x, y: x + y,
                     'optimizer_selector': DummySelector([OptimizerTest1]),
                     'bounds': ((0, 1), (0, 1)),
                     'overwrite_existing': True},
                  **kwargs}
        GloMPOManager(backend=backend, **kwargs)

    @pytest.mark.parametrize("summary_files", [0, -5, 10, 23.56, 2.3])
    def test_init_clipping(self, summary_files, backend):
        opt = GloMPOManager(task=lambda x, y: x + y,
                            optimizer_selector=DummySelector([OptimizerTest1]),
                            bounds=((0, 1), (0, 1)),
                            overwrite_existing=True,
                            summary_files=summary_files,
                            backend=backend)
        assert opt.summary_files == int(np.clip(int(summary_files), 0, 3))

    def test_init_workingdir(self, backend):
        with pytest.warns(UserWarning, match="Cannot parse working_dir"):
            GloMPOManager(task=lambda x, y: x + y,
                          optimizer_selector=DummySelector([OptimizerTest1]),
                          bounds=((0, 1), (0, 1)),
                          working_dir=5,
                          overwrite_existing=True,
                          backend=backend)

    def test_task_wrapper(self, backend):
        def task(x, *args, calculate=False):
            if calculate:
                return sum([*args]) ** x
            return None

        wrapped_task = task_args_wrapper(task, (1, 2, 3, 4), {'calculate': True})

        for i in range(5):
            assert wrapped_task(i) == 10 ** i

    def test_redirect(self, backend):
        def func():
            print("redirect_test")
            raise RuntimeError("redirect_test_error")

        wrapped_func = process_print_redirect(1, func)
        p = mp.Process(target=wrapped_func)
        p.start()
        p.join()

        with open("glompo_optimizer_printstreams/1_printstream.out", "r") as file:
            assert file.readline() == "redirect_test\n"

        with open("glompo_optimizer_printstreams/1_printstream.err", "r") as file:
            assert any(["redirect_test_error" in line for line in file.readlines()])

    def test_no_messaging(self, backend):
        manager = GloMPOManager(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                                optimizer_selector=CycleSelector([SilentOptimizer]),
                                bounds=((0, 1), (0, 1), (0, 1)),
                                working_dir="tests/temp_outputs",
                                overwrite_existing=True,
                                max_jobs=1,
                                summary_files=3,
                                convergence_checker=MaxOptsStarted(2),
                                backend=backend)
        with pytest.warns(RuntimeWarning, match="terminated normally without sending a"):
            manager.start_manager()

        with open("tests/temp_outputs/glompo_optimizer_logs/1_SilentOptimizer.yml", 'r') as stream:
            data = yaml.safe_load(stream)
            assert "Approximate Stop Time" in data['DETAILS']
            assert data['DETAILS']['End Condition'] == "Normal termination (Reason unknown)"

        sleep(1)  # Delay for process cleanup

    def test_messaging(self, backend):
        manager = GloMPOManager(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                                optimizer_selector=CycleSelector([MessagingOptimizer]),
                                bounds=((0, 1), (0, 1), (0, 1)),
                                working_dir="tests/temp_outputs",
                                overwrite_existing=True,
                                max_jobs=1,
                                summary_files=3,
                                convergence_checker=MaxOptsStarted(2),
                                backend=backend)
        with pytest.warns(None) as warns:
            manager.start_manager()
            for record in warns:
                assert "terminated normally without sending a" not in record.message

        with open("tests/temp_outputs/glompo_optimizer_logs/1_MessagingOptimizer.yml", 'r') as stream:
            data = yaml.safe_load(stream)
            assert "Approximate Stop Time" not in data['DETAILS']
            assert "Stop Time" in data['DETAILS']
            assert data['DETAILS']['End Condition'] != "Normal termination (Reason unknown)"
            assert "This is a test of the GloMPO signalling system" in data['MESSAGES']

        sleep(1)  # Delay for process cleanup

    def test_too_long_hangingopt(self, backend):

        if backend == 'threads':
            init_warning = pytest.warns(UserWarning, match="Cannot use force terminations with threading.")
        else:
            init_warning = pytest.warns(None)

        with init_warning:
            manager = GloMPOManager(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                                    optimizer_selector=CycleSelector([HangingOptimizer]),
                                    bounds=((0, 1), (0, 1), (0, 1)),
                                    working_dir="tests/temp_outputs",
                                    overwrite_existing=True,
                                    max_jobs=1,
                                    summary_files=3,
                                    convergence_checker=MaxOptsStarted(2),
                                    force_terminations_after=1,
                                    backend=backend)

        if backend == 'processes':
            with pytest.warns(RuntimeWarning, match="seems to be hanging. Forcing termination."):
                manager.start_manager()

            with open("tests/temp_outputs/glompo_optimizer_logs/1_HangingOptimizer.yml", 'r') as stream:
                data = yaml.safe_load(stream)
                assert "Approximate Stop Time" in data['DETAILS']
                assert data['DETAILS']['End Condition'] == "Forced GloMPO Termination"
                assert "Force terminated due to no feedback timeout." in data['MESSAGES']

            sleep(1)  # Delay for process cleanup

    def test_too_long_hangingterm(self, backend):

        if backend == 'threads':
            init_warning = pytest.warns(UserWarning, match="Cannot use force terminations with threading.")
        else:
            init_warning = pytest.warns(None)

        with init_warning:
            manager = GloMPOManager(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                                    optimizer_selector=CycleSelector([HangOnEndOptimizer]),
                                    bounds=((0, 1), (0, 1), (0, 1)),
                                    working_dir="tests/temp_outputs",
                                    overwrite_existing=True,
                                    max_jobs=2,
                                    enforce_elitism=True,
                                    summary_files=3,
                                    convergence_checker=MaxOptsStarted(3),
                                    killing_conditions=TrueHunter(2),
                                    force_terminations_after=1,
                                    split_printstreams=False,
                                    backend=backend)

        if backend == 'processes':
            with pytest.warns(RuntimeWarning, match="Forced termination signal sent to optimizer"):
                manager.start_manager()

            with open("tests/temp_outputs/glompo_optimizer_logs/2_HangOnEndOptimizer.yml", 'r') as stream:
                data = yaml.safe_load(stream)
                assert "Approximate Stop Time" in data['DETAILS']
                assert data['DETAILS']['End Condition'] == "Forced GloMPO Termination"
                assert "Force terminated due to no feedback after kill signal timeout." in data['MESSAGES']

                sleep(1)  # Delay for process cleanup

    def test_opt_error(self, backend):
        manager = GloMPOManager(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                                optimizer_selector=CycleSelector([ErrorOptimizer]),
                                bounds=((0, 1), (0, 1), (0, 1)),
                                working_dir="tests/temp_outputs",
                                overwrite_existing=True,
                                max_jobs=1,
                                summary_files=3,
                                convergence_checker=MaxOptsStarted(2),
                                backend=backend)

        with pytest.warns(RuntimeWarning, match="terminated in error"):
            manager.start_manager()

        with open("tests/temp_outputs/glompo_optimizer_logs/1_ErrorOptimizer.yml", 'r') as stream:
            data = yaml.safe_load(stream)
            assert "Approximate Stop Time" in data['DETAILS']
            assert "Error termination (exitcode" in data['DETAILS']['End Condition']
            assert any(["Terminated in error with code" in message for message in data['MESSAGES']])

            sleep(1)  # Delay for process cleanup

    def test_manager_error(self, backend):
        manager = GloMPOManager(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                                optimizer_selector=CycleSelector([SilentOptimizer]),
                                bounds=((0, 1), (0, 1), (0, 1)),
                                working_dir="tests/temp_outputs",
                                overwrite_existing=True,
                                max_jobs=1,
                                summary_files=1,
                                convergence_checker=ErrorChecker(),
                                backend=backend)
        with pytest.warns(RuntimeWarning, match="Optimization failed. Caught exception: "
                                                "This is a test of the GloMPO error management system."):
            manager.start_manager()

        with open("tests/temp_outputs/glompo_manager_log.yml", "r") as stream:
            data = yaml.safe_load(stream)
            assert "Process Crash" in data['Solution']['exit cond.']
            sleep(1)  # Delay for process cleanup

    @pytest.mark.mini
    def test_mwe(self, backend):
        class SteepestGradient(BaseOptimizer):

            needscaler = False

            def __init__(self, max_iters, gamma, precision, workers=1,
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
                                                              it > self.max_iters))
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

            def push_iter_result(self, iters):
                self._results_queue.put(iters)

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
            x, _ = pt
            calc = np.exp(-x ** 2 / 5000)
            calc *= x * np.cos(0.2 * x)
            calc /= 125000 * np.sqrt(2 * np.pi)
            calc += 0.00159577 * np.exp(-x ** 2 / 5000) * np.sin(0.2 * x)
            return calc

        def df_dy(pt):
            _, y = pt
            calc = 2e-6 * y
            return calc

        # x0_generator
        class IntervalGenerator(BaseGenerator):
            def __init__(self):
                super().__init__()
                self.count = -1

            def generate(self, *args, **kwargs) -> np.ndarray:
                self.count += 1
                x = np.random.choice([-1, 1]) * 30 * self.count + np.random.uniform(-2, 2)
                y = np.random.uniform(-100, 100)
                return x, y

        manager = GloMPOManager(task=f,
                                optimizer_selector=CycleSelector([(SteepestGradient, {'max_iters': 10000,
                                                                                      'precision': 1e-8,
                                                                                      'gamma': [100, 100000]}, None)]),
                                bounds=((-100, 100), (-100, 100)),
                                working_dir='tests/outputs',
                                overwrite_existing=True,
                                max_jobs=3,
                                task_kwargs={'delay': 0.1},
                                backend=backend,
                                convergence_checker=KillsAfterConvergence(2, 1) | MaxFuncCalls(10000) | MaxSeconds(
                                    5 * 60),
                                x0_generator=IntervalGenerator(),
                                killing_conditions=MinIterations(1000),
                                summary_files=3,
                                visualisation=False,
                                visualisation_args=None)
        result = manager.start_manager()
        assert np.all(np.isclose(result.x, 0, atol=1e-6))
        assert np.isclose(result.fx, -0.00797884560802864)
        assert result.origin['opt_id'] == 1
        assert result.origin['type'] == 'SteepestGradient'

    @classmethod
    def teardown_class(cls):
        os.chdir(cls.base_wd)
        try:
            shutil.rmtree("tests/temp_outputs", ignore_errors=True)
            shutil.rmtree("glompo_optimizer_printstreams", ignore_errors=True)
            is_save = '--save-outs' in sys.argv
            is_mini = '--run-minimize' in sys.argv
            if is_mini and not is_save:
                shutil.rmtree("tests/outputs", ignore_errors=True)
        except FileNotFoundError:
            pass
