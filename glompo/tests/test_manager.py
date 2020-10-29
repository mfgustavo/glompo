import logging
import multiprocessing as mp
import tarfile
from pathlib import Path
from time import sleep, time
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

import numpy as np
import pytest
import yaml

from glompo.common.helpers import CheckpointingError
from glompo.common.namedtuples import IterationResult, ProcessPackage, Result
from glompo.convergence import BaseChecker, KillsAfterConvergence, MaxFuncCalls, MaxOptsStarted, MaxSeconds
from glompo.core._backends import CustomThread
from glompo.core.checkpointing import CheckpointingControl
from glompo.core.manager import GloMPOManager
from glompo.core.optimizerlogger import OptimizerLogger
from glompo.generators import BaseGenerator, RandomGenerator
from glompo.hunters import BaseHunter, MinIterations
from glompo.opt_selectors import BaseSelector, CycleSelector, IterSpawnStop
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from glompo.optimizers.random import RandomOptimizer

""" Helper Classes """


class DummySelector(BaseSelector):
    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: OptimizerLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None]:
        pass


class OptimizerTest1(BaseOptimizer):
    signal_pipe = None
    results_queue = None

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        pass

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def checkpoint_save(self, *args):
        pass


class OptimizerTest2:
    pass


class MessagingOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        self.message_manager(9, "This is a test of the GloMPO signalling system")
        self.message_manager(0)
        sleep(1)

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def checkpoint_save(self, *args):
        pass


class SilentOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        sleep(1)

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def checkpoint_save(self, *args):
        pass


class HangingOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        while True:
            pass

    def push_iter_result(self, *args):
        pass

    def callstop(self, *args):
        pass

    def checkpoint_save(self, *args):
        pass


class HangOnEndOptimizer(BaseOptimizer):

    def __init__(self, opt_id: int = None, signal_pipe=None, results_queue=None, pause_flag=None, workers=1,
                 backend='processes'):
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)
        self.constant = opt_id

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        i = 0
        while True:
            i += 1
            self.check_messages()
            sleep(0.1)
            self.push_iter_result(
                IterationResult(self._opt_id, i, 1, [i], np.sin(i / (2 * np.pi)) + self.constant, False))

    def push_iter_result(self, ir):
        self._results_queue.put(ir)

    def callstop(self, *args):
        print("Hanging Callstop Activated")
        while True:
            pass

    def checkpoint_save(self, *args):
        pass


class ErrorOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        raise RuntimeError("This is a test of the GloMPO error handling service")

    def push_iter_result(self, ir):
        pass

    def callstop(self, *args):
        pass

    def checkpoint_save(self, *args):
        pass


class TrueHunter(BaseHunter):
    def __init__(self, target: int):
        super().__init__()
        self.target = target

    def __call__(self, log, hunter_opt_id, victim_opt_id) -> bool:
        return victim_opt_id == self.target


class ErrorChecker(BaseChecker):
    def __init__(self, err):
        super().__init__()
        self.err = err

    def __call__(self, manager: 'GloMPOManager') -> bool:
        raise self.err("This is a test of the GloMPO error management system.")


class TrueChecker(BaseChecker):
    def __call__(self, *args, **kwargs):
        return True


class LogBlocker:
    """ Pytest automatically captures logs. This makes GloMPO checkpointing since the loggers cannot be pickled within
        a pytest run. LogBlocker replaces traditional Python Loggers with a blank object allowing checkoints.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def addHandler(self, *args, **kwrags):
        pass

    def removeHandler(self, *args, **kwargs):
        pass


""" Module Fixtures """


@pytest.fixture()
def manager():
    return GloMPOManager()


@pytest.fixture(scope='function')
def mask_psutil():
    import glompo.core.manager
    original = glompo.core.manager.HAS_PSUTIL
    glompo.core.manager.HAS_PSUTIL = False
    yield
    glompo.core.manager.HAS_PSUTIL = original


@pytest.fixture(scope='function')
def mask_dill():
    import glompo.core.manager
    original = glompo.core.manager.HAS_DILL
    glompo.core.manager.HAS_DILL = False
    yield
    glompo.core.manager.HAS_DILL = original


@pytest.fixture(scope='function')
def hanging_process():
    def child_process(*args, **kwargs):
        while True:
            pass

    process = mp.Process(target=child_process)
    process.start()

    yield process

    if process.is_alive():
        process.terminate()
    process.join()


""" Module Tests"""


class TestManager:

    @pytest.mark.parametrize("kwargs", [{'task': None},
                                        {'opt_selector': {'default': OptimizerTest2}},
                                        {'opt_selector': OptimizerTest1()},
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
    def test_init_typeerr(self, kwargs, manager):
        with pytest.raises(TypeError):
            keys = {**{'task': lambda x, y: x + y,
                       'opt_selector': DummySelector([OptimizerTest1]),
                       'bounds': ((0, 1), (0, 1)),
                       'overwrite_existing': True},
                    **kwargs}
            manager.setup(**keys)

    @pytest.mark.parametrize("kwargs", [{'bounds': ((0, 0), (0, 1))},
                                        {'bounds': ((1, 0), (0, 1))},
                                        {'max_jobs': -1}])
    def test_init_valerr(self, kwargs, manager):
        with pytest.raises(ValueError):
            keys = {**{'task': lambda x, y: x + y,
                       'opt_selector': DummySelector([OptimizerTest1]),
                       'bounds': ((0, 1), (0, 1)),
                       'overwrite_existing': True},
                    **kwargs}
            manager.setup(**keys)

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_invalid_backend(self, backend, manager):
        with pytest.warns(UserWarning, match="Unable to parse backend"):
            keys = {'task': lambda x, y: x + y,
                    'opt_selector': DummySelector([OptimizerTest1]),
                    'bounds': ((0, 1), (0, 1)),
                    'overwrite_existing': True}
            manager.setup(backend='magic', **keys)

    @pytest.mark.parametrize("kwargs", [{},
                                        {'x0_generator': RandomGenerator(((0, 1), (0, 1)))},
                                        {'convergence_checker': KillsAfterConvergence()},
                                        {'max_jobs': 3},
                                        {'killing_conditions': MinIterations(10)}])
    def test_init(self, kwargs):
        kwargs = {**{'task': lambda x, y: x + y,
                     'opt_selector': DummySelector([OptimizerTest1]),
                     'bounds': ((0, 1), (0, 1)),
                     'overwrite_existing': True},
                  **kwargs}
        manager = GloMPOManager.new_manager(**kwargs)
        assert manager.is_initialised

    @pytest.mark.parametrize("summary_files", [0, -5, 10, 23.56, 2.3])
    def test_init_clipping(self, summary_files, manager):
        manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                      opt_selector=DummySelector([OptimizerTest1]), overwrite_existing=True,
                      summary_files=summary_files)
        assert manager.summary_files == int(np.clip(int(summary_files), 0, 5))

    def test_init_workingdir(self, manager):
        with pytest.warns(UserWarning, match="Cannot parse working_dir"):
            manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                          opt_selector=DummySelector([OptimizerTest1]), working_dir=5, overwrite_existing=True)

    def test_init_block_checkpointing(self, manager, mask_dill):
        with pytest.warns(UserWarning, match="Checkpointing controls ignored. Cannot setup infrastructure without "):
            manager.setup(task=lambda x, y: x / 0, bounds=((0, 1), (0, 1)),
                          opt_selector=DummySelector([OptimizerTest1]), checkpoint_control=CheckpointingControl())
        assert manager.checkpoint_control is None

    @pytest.mark.parametrize('init, kwargs', [(GloMPOManager.setup,
                                               {'task': None, 'bounds': None, 'opt_selector': None}),
                                              (GloMPOManager.load_checkpoint,
                                               {'path': None})])
    def test_double_init(self, manager, init, kwargs):
        manager._is_restart = "Changes value of is_initialised"
        with pytest.warns(UserWarning, match="Manager already initialised, cannot reinitialise. Aborting"):
            init(manager, **kwargs)

    def test_not_init(self, manager):
        with pytest.warns(UserWarning, match="Cannot start manager, initialise manager first with setup or"):
            manager.start_manager()

    def test_overwrite(self, tmp_path, manager, mask_psutil):
        for folder in ("cmadata", "glompo_optimizer_logs", "glompo_optimizer_printstreams"):
            (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        for file in ("glompo_manager_log.yml", "trajectories.png", "trajectories_log_best.png", "opt123_parms.png"):
            (tmp_path / file).touch()

        manager.setup(task=lambda x, y: x / 0, bounds=((0, 1), (0, 1)),
                      opt_selector=DummySelector([OptimizerTest1]), working_dir=tmp_path,
                      convergence_checker=TrueChecker(), overwrite_existing=True, summary_files=0,
                      split_printstreams=False, visualisation=False)

        manager._purge_old_results()

        assert [*tmp_path.iterdir()] == [tmp_path / "cmadata"]

    @pytest.mark.parametrize("workers", [1, 3, 6])
    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_opt_slot_filling(self, workers, backend, monkeypatch, manager, tmp_path):

        class FakeProcess:
            def is_alive(self):
                return True

        def mock_start_job(opt_id, optimizer, call_kwargs, pipe, event, workers):
            manager._optimizer_packs[opt_id] = ProcessPackage(FakeProcess(), pipe, event, workers)

        manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                      opt_selector=CycleSelector([(OptimizerTest1, {'workers': workers}, None)]),
                      working_dir=tmp_path, max_jobs=10, backend=backend, split_printstreams=False)

        monkeypatch.setattr(manager, "_start_new_job", mock_start_job)

        manager._fill_optimizer_slots()

        assert len(manager._optimizer_packs) == int(10 / workers)

    @pytest.mark.parametrize("fcalls", [0, 3, 6, 10])
    def test_spawning_stop(self, fcalls, manager, tmp_path):
        manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                      opt_selector=CycleSelector([OptimizerTest1], allow_spawn=IterSpawnStop(5)),
                      working_dir=tmp_path, max_jobs=10, split_printstreams=False)
        manager.f_counter = fcalls
        opt = manager._setup_new_optimizer(1)
        if fcalls <= 5:
            assert opt
            assert manager.spawning_opts
        else:
            assert opt is None
            assert not manager.spawning_opts

    def test_filexerr(self, tmp_path, manager):
        (tmp_path / "glompo_manager_log.yml").touch()

        manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)), working_dir=tmp_path,
                      opt_selector=DummySelector([OptimizerTest1]), overwrite_existing=False,
                      split_printstreams=False)

        with pytest.raises(FileExistsError):
            manager.start_manager()

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_no_messaging(self, backend, manager, tmp_path):
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector([SilentOptimizer]), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=3, split_printstreams=False)
        with pytest.warns(RuntimeWarning, match="terminated normally without sending a"):
            manager.start_manager()

        with Path(tmp_path, "glompo_optimizer_logs", "1_SilentOptimizer.yml").open('r') as stream:
            data = yaml.safe_load(stream)
            assert "Approximate Stop Time" in data['DETAILS']
            assert data['DETAILS']['End Condition'] == "Normal termination (Reason unknown)"

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_messaging(self, backend, manager, tmp_path):
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector([MessagingOptimizer]), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=3)
        with pytest.warns(None) as warns:
            manager.start_manager()
            for record in warns:
                assert "terminated normally without sending a" not in record.message

        with Path(tmp_path, "glompo_optimizer_logs", "1_MessagingOptimizer.yml").open('r') as stream:
            data = yaml.safe_load(stream)
            assert "Approximate Stop Time" not in data['DETAILS']
            assert "Stop Time" in data['DETAILS']
            assert data['DETAILS']['End Condition'] != "Normal termination (Reason unknown)"
            assert "This is a test of the GloMPO signalling system" in data['MESSAGES']

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_too_long_hangingopt(self, backend, manager, tmp_path):

        if backend == 'threads':
            run_warning = pytest.warns(UserWarning, match="Cannot use force terminations with threading.")
        else:
            run_warning = pytest.warns(RuntimeWarning, match="seems to be hanging. Forcing termination.")

        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                      bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector([HangingOptimizer]), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=3, force_terminations_after=1)

        if backend == 'processes':
            with run_warning:
                manager.start_manager()

            with Path(tmp_path, "glompo_optimizer_logs", "1_HangingOptimizer.yml").open('r') as stream:
                data = yaml.safe_load(stream)
                assert "Approximate Stop Time" in data['DETAILS']
                assert data['DETAILS']['End Condition'] == "Forced GloMPO Termination"
                assert "Force terminated due to no feedback timeout." in data['MESSAGES']

    def test_too_long_hangingterm(self, manager, tmp_path, hanging_process, mask_psutil):
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                      bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector([HangOnEndOptimizer]), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=2, backend='processes',
                      convergence_checker=MaxOptsStarted(3), killing_conditions=TrueHunter(2),
                      summary_files=3, force_terminations_after=30, split_printstreams=False)

        manager._optimizer_packs[1] = ProcessPackage(hanging_process, None, None, 1)
        manager.hunt_victims[1] = time() - 30
        manager._last_feedback[1] = time() - 30
        manager.opt_log.add_optimizer(1, None, time() - 60)

        with pytest.warns(RuntimeWarning, match="Forced termination signal sent to optimizer 1."):
            manager._inspect_children()

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_opt_error(self, backend, manager, tmp_path):
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector([ErrorOptimizer]), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=3, split_printstreams=False)

        with pytest.warns(RuntimeWarning, match="terminated in error"):
            manager.start_manager()

        with Path(tmp_path, "glompo_optimizer_logs", "1_ErrorOptimizer.yml").open('r') as stream:
            data = yaml.safe_load(stream)
            assert "Approximate Stop Time" in data['DETAILS']
            assert "Error termination (exitcode" in data['DETAILS']['End Condition']
            assert any(["Terminated in error with code" in message for message in data['MESSAGES']])

    @pytest.mark.parametrize("err", [RuntimeError, KeyboardInterrupt])
    def test_manager_error(self, err, manager, tmp_path, mask_psutil, monkeypatch):
        def mock_fill_slots(*args, **kwargs):
            raise err

        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector([SilentOptimizer]), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, summary_files=1, split_printstreams=False)
        if err == RuntimeError:
            match = "Optimization failed. Caught exception: "
            reason = "Process Crash"
        else:
            match = "Optimization failed. Caught User Interrupt"
            reason = "User Interrupt"

        monkeypatch.setattr(manager, '_fill_optimizer_slots', mock_fill_slots)

        with pytest.warns(RuntimeWarning, match=match):
            manager.start_manager()

        with Path(tmp_path, "glompo_manager_log.yml").open('r') as stream:
            data = yaml.safe_load(stream)
            print(data)
            assert reason in data['Solution']['exit cond.']

    @pytest.mark.parametrize('backend', ['processes', 'processes_forced', 'threads'])
    def test_backend_prop(self, backend, tmp_path):
        if backend == 'threads':
            opt_type = CustomThread
            opt_backend = 'threads'
            is_daemon = True
        elif backend == 'processes':
            opt_type = mp.Process
            opt_backend = 'threads'
            is_daemon = True
        elif backend == 'processes_forced':
            opt_type = mp.Process
            opt_backend = 'processes'
            is_daemon = False

        manager = GloMPOManager()
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                      bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector([SilentOptimizer]), working_dir=tmp_path,
                      overwrite_existing=True, backend=backend, summary_files=0, split_printstreams=False)
        opt_pack = manager._setup_new_optimizer(1)
        assert opt_pack.optimizer._backend == opt_backend

        manager._start_new_job(*opt_pack)
        assert manager._optimizer_packs[1].process.daemon == is_daemon
        assert type(manager._optimizer_packs[1].process) is opt_type

    @pytest.mark.parametrize("fx, is_log", [(range(1000, 10), False),
                                            (range(10000, 100), False),
                                            (range(1000, 10002000, 1000), True),
                                            (range(-100, 100), False),
                                            (range(-10, 100000, 100), False)])
    def test_plot_construction(self, monkeypatch, fx, is_log, manager, tmp_path, mask_psutil):

        gathered = [is_log]

        def mock_save_traj(name, log_scale, best_fx):
            gathered.append(log_scale)

        def pass_meth(*args):
            pass

        manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                      opt_selector=CycleSelector([OptimizerTest1]), working_dir=tmp_path, max_jobs=10,
                      summary_files=4, split_printstreams=False)

        monkeypatch.setattr(manager.opt_log, "plot_trajectory", mock_save_traj)
        monkeypatch.setattr(manager.opt_log, "save_summary", pass_meth)
        monkeypatch.setattr(manager.opt_log, "save_optimizer", pass_meth)

        manager.o_counter = 1
        manager.opt_log.add_optimizer(1, OptimizerTest1.__name__, 0)
        for i, f in enumerate(fx):
            manager.opt_log.put_iteration(1, i, i, i, [0.5, 0.5], float(f))

        manager._save_log(Result([0.2, 0.3], 65.54, {}, {}), "GloMPO Convergence", "", tmp_path, 5)

        assert len(set(gathered)) == 1

    @pytest.mark.mini
    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_mwe(self, backend, manager, save_outputs, tmp_path):
        class SteepestGradient(BaseOptimizer):

            def __init__(self, max_iters, gamma, precision, opt_id: int = None, signal_pipe=None,
                         results_queue=None, pause_flag=None, workers: int = 1,
                         backend: str = 'threads'):
                super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)
                self.max_iters = max_iters
                self.gamma = np.array(gamma)
                self.precision = precision
                self.terminate = False
                self.reason = None
                self.current_x = None
                self.result = MinimizeResult()

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
                    self.message_manager((0, self.reason))
                print(f"Stopping due to {self.reason}")
                return self.result

            def callstop(self, *args):
                self.terminate = True
                self.reason = "manager termination"

        # Task
        def f(pt, delay=0.1):
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

        manager.setup(task=f, bounds=((-100, 100), (-100, 100)),
                      opt_selector=CycleSelector([(SteepestGradient, {'max_iters': 10000,
                                                                      'precision': 1e-8,
                                                                      'gamma': [100, 100000]}, None)]),
                      working_dir=tmp_path, overwrite_existing=True, max_jobs=3, backend=backend,
                      convergence_checker=KillsAfterConvergence(2, 1) | MaxFuncCalls(10000) | MaxSeconds(
                          session_max=60),
                      x0_generator=IntervalGenerator(), killing_conditions=MinIterations(1000),
                      summary_files=5, visualisation=False, visualisation_args=None)
        result = manager.start_manager()
        assert np.all(np.isclose(result.x, 0, atol=1e-6))
        assert np.isclose(result.fx, -0.00797884560802864)
        assert result.origin['opt_id'] == 1
        assert result.origin['type'] == 'SteepestGradient'


class TestCheckpointing:
    """ Tests related to checkpointing and resuming GloMPO optimizations. These tests rely on a single shared directory.
        The first tests produce checkpoints (at init, midway and convergence) and subsequent tests attempt to resume
        from these.
    """

    @pytest.fixture(scope='function')
    def mnkyptch_loggers(self, monkeypatch):
        monkeypatch.setattr(logging, 'getLogger', LogBlocker)

    def test_init_save(self, manager, tmp_path, mnkyptch_loggers, save_outputs, request):
        request.config.cache.set('init_checkpoint', None)
        manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                      bounds=[(0, 100)] * 3,
                      opt_selector=CycleSelector([(RandomOptimizer, {'iters': 1000}, None)]),
                      working_dir=tmp_path,
                      max_jobs=2,
                      backend='processes',
                      convergence_checker=None,
                      x0_generator=None,
                      killing_conditions=None,
                      checkpoint_control=CheckpointingControl(checkpoint_at_init=True, naming_format='chkpt_%(count)',
                                                              checkpointing_dir=tmp_path),
                      visualisation=True,
                      visualisation_args={'x_range': 2000})
        assert (tmp_path / 'chkpt_000.tar.gz').exists()
        assert not (tmp_path / 'chkpt_000').exists()
        with tarfile.open(tmp_path / 'chkpt_000.tar.gz', 'r:gz') as tfile:
            members = tfile.getnames()
            assert 'task' in members
            assert 'scope' in members
            assert 'manager' in members
            assert 'optimizers' in members
            assert all(['optimizers/' not in member for member in members])
        request.config.cache.set('init_checkpoint', str(tmp_path / 'chkpt_000.tar.gz'))

    def test_init_load(self, tmp_path, mnkyptch_loggers, save_outputs, request):
        checkpoint_path = request.config.cache.get('init_checkpoint', None)
        if not checkpoint_path or not Path(checkpoint_path).exists():
            pytest.xfail(reason="Checkpoint not found")

        manager = GloMPOManager()
        manager.load_checkpoint(path=checkpoint_path,
                                working_dir=tmp_path,
                                summary_files=1,
                                convergence_checker=MaxSeconds(session_max=3),
                                backend='threads',
                                force_terminations_after=60,
                                visualisation_args={'x_range': (0, 4000)})

        assert manager.task([0.2, 4, 6.1]) == 0.2 ** 2 + 3 * 4 ** 4 - 6.1 ** 0.5
        assert manager.working_dir == tmp_path
        assert manager.max_jobs == 2
        assert manager.f_counter == 0
        assert manager.o_counter == 0
        assert manager.visualisation
        assert manager._optimizer_packs == {}
        assert manager.t_used == 0
        assert manager.summary_files == 1
        assert manager.t_start is None
        assert manager.opt_crashed is False
        assert manager.last_opt_spawn == (0, 0)
        assert manager._too_long == 60
        assert manager.allow_forced_terminations
        assert manager.visualisation_args == {'x_range': 2000}
        assert not manager.proc_backend

        with pytest.warns(UserWarning, match="Cannot use force terminations with threading."):
            manager.start_manager()
        assert (tmp_path / 'glompo_manager_log.yml').exists()

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_mdpt_save(self, manager, tmp_path, mnkyptch_loggers, save_outputs, request, backend):
        request.config.cache.set('mdpt_checkpoint', None)
        manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                      bounds=[(0, 100)] * 3,
                      opt_selector=CycleSelector([(RandomOptimizer, {'iters': 1000}, None)]),
                      working_dir=tmp_path,
                      max_jobs=2,
                      backend=backend,
                      convergence_checker=MaxSeconds(session_max=3),
                      x0_generator=None,
                      killing_conditions=None,
                      checkpoint_control=CheckpointingControl(checkpoint_time_frequency=2,
                                                              naming_format='chkpt_%(count)',
                                                              checkpointing_dir=tmp_path),
                      visualisation=True,
                      visualisation_args={'record_movie': True, 'movie_kwargs': {'outfile': tmp_path / 'mv.mp4'}})
        with pytest.warns(RuntimeWarning, match="Movie saving is not supported"):
            manager.start_manager()

        assert manager.f_counter > 0
        assert manager.o_counter > 0
        assert manager.t_end - manager.t_start - 3 < 1
        assert manager.result.fx

        assert (tmp_path / 'chkpt_000.tar.gz').exists()
        assert not (tmp_path / 'chkpt_000').exists()
        with tarfile.open(tmp_path / 'chkpt_000.tar.gz', 'r:gz') as tfile:
            members = tfile.getnames()
            assert 'task' in members
            assert 'scope' in members
            assert 'manager' in members
            assert 'optimizers' in members
            assert any(['optimizers/' in member for member in members])
        request.config.cache.set('mdpt_checkpoint', str(tmp_path / 'chkpt_000.tar.gz'))

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_mdpt_load(self, tmp_path, mnkyptch_loggers, save_outputs, request, backend):
        checkpoint_path = request.config.cache.get('mdpt_checkpoint', None)
        if not checkpoint_path or not Path(checkpoint_path).exists():
            pytest.xfail(reason="Checkpoint not found")

        manager = GloMPOManager.load_manager(path=checkpoint_path,
                                             working_dir=tmp_path,
                                             backend=backend,
                                             convergence_checker=MaxSeconds(overall_max=5),
                                             checkpointing_control=None,
                                             summary_files=1)

        init_f_count = manager.f_counter
        assert manager.task([0.2, 4, 6.1]) == 0.2 ** 2 + 3 * 4 ** 4 - 6.1 ** 0.5
        assert manager.working_dir == tmp_path
        assert manager.max_jobs == 2
        assert init_f_count > 0
        assert manager.o_counter == 2
        assert manager.visualisation
        assert 2 < manager.t_used < 2.6
        assert manager.summary_files == 1
        assert manager.t_start is None
        assert manager.opt_crashed is False
        assert manager.last_opt_spawn == (0, 0)

        with pytest.warns(RuntimeWarning, match="Movie saving is not supported"):
            manager.start_manager()
        assert (tmp_path / 'glompo_manager_log.yml').exists()
        assert manager.f_counter > init_f_count
        assert manager.t_end - manager.t_start - 3 < 1

    def test_conv_save(self, manager, tmp_path, mnkyptch_loggers, save_outputs, request):
        manager: GloMPOManager
        request.config.cache.set('conv_checkpoint', None)
        manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                      bounds=[(0, 100)] * 3,
                      opt_selector=CycleSelector([(RandomOptimizer, {'iters': 100000}, None)]),
                      working_dir=tmp_path,
                      max_jobs=2,
                      backend='processes',
                      convergence_checker=MaxSeconds(overall_max=3),
                      x0_generator=None,
                      killing_conditions=None,
                      checkpoint_control=CheckpointingControl(checkpoint_at_conv=True,
                                                              naming_format='chkpt_%(count)',
                                                              checkpointing_dir=tmp_path),
                      visualisation=False,
                      summary_files=4)
        manager.start_manager()

        assert manager.f_counter > 0
        assert manager.o_counter > 0
        assert manager.t_end - manager.t_start - 3 < 1.1
        assert manager.result.fx

        assert (tmp_path / 'chkpt_000.tar.gz').exists()
        assert not (tmp_path / 'chkpt_000').exists()
        with tarfile.open(tmp_path / 'chkpt_000.tar.gz', 'r:gz') as tfile:
            members = tfile.getnames()
            assert 'task' in members
            assert 'manager' in members
            assert 'optimizers' in members
            assert any(['optimizers/' in member for member in members])
        request.config.cache.set('conv_checkpoint', str(tmp_path / 'chkpt_000.tar.gz'))

    def test_conv_load(self, tmp_path, mnkyptch_loggers, request):
        checkpoint_path = request.config.cache.get('conv_checkpoint', None)
        if not checkpoint_path or not Path(checkpoint_path).exists():
            pytest.xfail(reason="Checkpoint not found")

        with pytest.warns(RuntimeWarning, match="The convergence criteria already evaluates to True. The manager will "
                                                "be unable to resume"):
            manager = GloMPOManager.load_manager(path=checkpoint_path,
                                                 working_dir=tmp_path,
                                                 summary_files=1)

        init_f_count = manager.f_counter
        init_result = manager.result
        init_dt_starts = manager.dt_starts
        init_dt_ends = manager.dt_ends

        assert manager.task([0.2, 4, 6.1]) == 0.2 ** 2 + 3 * 4 ** 4 - 6.1 ** 0.5
        assert manager.working_dir == tmp_path
        assert manager.max_jobs == 2
        assert init_f_count > 0
        assert manager.o_counter == 2
        assert not manager.visualisation
        assert 3 < manager.t_used < 3.5
        assert manager.summary_files == 1
        assert manager.t_start is None
        assert manager.opt_crashed is False
        assert manager.last_opt_spawn == (0, 0)

        with pytest.warns(RuntimeWarning, match="Convergence conditions met before optimizer start"):
            manager.start_manager()
        assert manager.f_counter == init_f_count
        assert manager.result == init_result
        assert manager.dt_starts == init_dt_starts
        assert manager.dt_ends == init_dt_ends

    def test_load_taskloader(self, manager, input_files, caplog):
        caplog.set_level(logging.INFO, logger='glompo.manager')

        def mock_taskloader(path):
            assert isinstance(path, str) or isinstance(path, Path)
            return lambda x: x[0] + x[1] + x[2]

        manager.load_checkpoint(input_files / 'no_task_chkpt.tar.gz', task_loader=mock_taskloader)
        assert manager.task([4, 8, 8]) == 20
        assert "No task detected in checkpoint, task or task_loader required." in caplog.messages
        assert "Task successfully loaded." in caplog.messages

    def test_load_newtask(self, input_files, manager, caplog):
        manager.load_checkpoint(input_files / 'no_task_chkpt.tar.gz', task=lambda x: x[0] + x[1] + x[2])
        assert manager.task([4, 8, 8]) == 20
        assert "No task detected in checkpoint, task or task_loader required." in caplog.messages

    def test_new_maxjobs(self, input_files, manager):
        with pytest.raises(CheckpointingError, match="Insufficient max_jobs allowed to restart all optimizers in"):
            with pytest.warns(UserWarning, match="The maximum number of jobs allowed is less than that demanded by "):
                manager.load_checkpoint(input_files / 'mock_chkpt.tar.gz', max_jobs=1)
