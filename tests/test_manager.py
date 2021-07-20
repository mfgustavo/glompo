import logging
import multiprocessing as mp
import shutil
import tarfile
from contextlib import contextmanager
from pathlib import Path
from time import sleep, time
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

import numpy as np
import pytest
import tables as tb
import yaml

try:
    import dill

    HAS_DILL = True
except (ModuleNotFoundError, ImportError):
    HAS_DILL = False

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader as Loader

from glompo.common.helpers import CheckpointingError
from glompo.common.namedtuples import Bound, IterationResult, ProcessPackage, Result
from glompo.convergence import BaseChecker, KillsAfterConvergence, MaxFuncCalls, MaxOptsStarted, MaxSeconds
from glompo.core._backends import CustomThread
from glompo.core.checkpointing import CheckpointingControl
from glompo.core.manager import GloMPOManager
from glompo.core.optimizerlogger import BaseLogger
from glompo.generators import BaseGenerator, RandomGenerator
from glompo.hunters import BaseHunter, MinFuncCalls, TypeHunter
from glompo.opt_selectors import BaseSelector, CycleSelector
from glompo.opt_selectors.spawncontrol import IterSpawnStop
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from glompo.optimizers.random import RandomOptimizer

""" Helper Classes """


class DummySelector(BaseSelector):
    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: BaseLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None]:
        pass


class OptimizerTest1(BaseOptimizer):
    signal_pipe = None
    results_queue = None

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        pass

    def callstop(self, *args):
        pass


class OptimizerTest2:
    pass


class MessagingOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        self.message_manager(9, "This is a test of the GloMPO signalling system")
        self.message_manager(0)
        sleep(1)

    def callstop(self, *args):
        pass


class SilentOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        sleep(1)

    def callstop(self, *args):
        pass


class HangingOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        while True:
            pass

    def callstop(self, *args):
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

    def callstop(self, *args):
        print("Hanging Callstop Activated")
        while True:
            pass


class ErrorOptimizer(BaseOptimizer):

    def minimize(self, function: Callable, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        raise RuntimeError("This is a test of the GloMPO error handling service")

    def callstop(self, *args):
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
    """ Pytest automatically captures logs. This makes GloMPO checkpointing impossible since the loggers cannot be
        pickled within a pytest run. LogBlocker replaces traditional Python Loggers with a blank object allowing
        checkpoints.
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

    def isEnabledFor(self, *args, **kwargs):
        return True

    @property
    def handlers(self):
        return []


""" Module Fixtures """


@contextmanager
def does_not_raise():
    yield


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


class TestManagement:

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
                                        {'killing_conditions': MinFuncCalls},
                                        {'killing_conditions': OptimizerTest1()},
                                        {'killing_conditions': OptimizerTest1},
                                        {'task_args': 564},
                                        {'task_kwargs': 66},
                                        {'gpr_training': 200}
                                        ])
    def test_init_typeerr(self, kwargs, manager):
        with pytest.raises(TypeError):
            keys = {**{'task': lambda x, y: x + y,
                       'opt_selector': DummySelector(OptimizerTest1),
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
                       'opt_selector': DummySelector(OptimizerTest1),
                       'bounds': ((0, 1), (0, 1)),
                       'overwrite_existing': True},
                    **kwargs}
            manager.setup(**keys)

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_invalid_backend(self, backend, manager):
        with pytest.warns(UserWarning, match="Unable to parse backend"):
            keys = {'task': lambda x, y: x + y,
                    'opt_selector': DummySelector(OptimizerTest1),
                    'bounds': ((0, 1), (0, 1)),
                    'overwrite_existing': True}
            manager.setup(backend='magic', **keys)

    @pytest.mark.parametrize("kwargs", [{},
                                        {'x0_generator': RandomGenerator(((0, 1), (0, 1)))},
                                        {'convergence_checker': KillsAfterConvergence()},
                                        {'max_jobs': 3},
                                        {'killing_conditions': MinFuncCalls(10)}])
    def test_init(self, kwargs):
        kwargs = {**{'task': lambda x, y: x + y,
                     'opt_selector': DummySelector(OptimizerTest1),
                     'bounds': ((0, 1), (0, 1)),
                     'overwrite_existing': True},
                  **kwargs}
        manager = GloMPOManager.new_manager(**kwargs)
        assert manager.is_initialised

    def test_init_workingdir(self, manager):
        with pytest.warns(UserWarning, match="Cannot parse working_dir"):
            manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                          opt_selector=DummySelector(OptimizerTest1), working_dir=5, overwrite_existing=True)

    def test_init_block_checkpointing(self, manager, mask_dill):
        with pytest.warns(ResourceWarning,
                          match="Checkpointing controls ignored. Cannot setup infrastructure without "):
            manager.setup(task=lambda x, y: x / 0, bounds=((0, 1), (0, 1)),
                          opt_selector=DummySelector(OptimizerTest1), checkpoint_control=CheckpointingControl())
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
        for folder in ("cmadata", "glompo_optimizer_printstreams"):
            (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        for file in ("glompo_manager_log.yml", "trajectories.png", "trajectories_log_best.png", "opt123_parms.png"):
            (tmp_path / file).touch()

        manager.setup(task=lambda x, y: x / 0, bounds=((0, 1), (0, 1)),
                      opt_selector=DummySelector(OptimizerTest1), working_dir=tmp_path,
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
                      opt_selector=CycleSelector((OptimizerTest1, {'workers': workers}, None)),
                      working_dir=tmp_path, max_jobs=10, backend=backend, split_printstreams=False)

        monkeypatch.setattr(manager, "_start_new_job", mock_start_job)

        manager._fill_optimizer_slots()

        assert len(manager._optimizer_packs) == int(10 / workers)

    @pytest.mark.parametrize("fcalls", [0, 3, 6, 10])
    def test_spawning_stop(self, fcalls, manager, tmp_path):
        manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                      opt_selector=CycleSelector(OptimizerTest1, allow_spawn=IterSpawnStop(5)),
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
                      opt_selector=DummySelector(OptimizerTest1), overwrite_existing=False,
                      split_printstreams=False)

        with pytest.raises(FileExistsError):
            manager.start_manager()

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_no_messaging(self, backend, manager, tmp_path, monkeypatch):
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector(SilentOptimizer), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=2, split_printstreams=False)

        with pytest.warns(RuntimeWarning, match="terminated normally without sending a"):
            manager.start_manager()

        assert 't_stop' in manager.opt_log._storage[1]['metadata']
        assert manager.opt_log._storage[1]['metadata']['end_cond'] == "Normal termination (Reason unknown)"

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_messaging(self, backend, manager, tmp_path, monkeypatch):
        def mock(*args, **kwargs): ...

        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector(MessagingOptimizer), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=2)

        monkeypatch.setattr(manager.opt_log, 'plot_trajectory', mock)
        monkeypatch.setattr(manager.opt_log, 'plot_optimizer_trials', mock)

        with pytest.warns(None) as warns:
            manager.start_manager()
            for record in warns:
                assert "terminated normally without sending a" not in record.message

        assert 't_stop' in manager.opt_log._storage[1]['metadata']
        assert manager.opt_log._storage[1]['metadata']['end_cond'] != "Normal termination (Reason unknown)"
        assert "This is a test of the GloMPO signalling system" in manager.opt_log._storage[1]['messages']

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_too_long_hangingopt(self, backend, manager, tmp_path, monkeypatch):
        def mock(*args, **kwargs):
            ...

        if backend == 'threads':
            run_warning = pytest.warns(UserWarning, match="Cannot use force terminations with threading.")
        else:
            run_warning = pytest.warns(RuntimeWarning, match="seems to be hanging. Forcing termination.")

        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                      bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector(HangingOptimizer), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=2, force_terminations_after=1)

        monkeypatch.setattr(manager.opt_log, 'plot_trajectory', mock)
        monkeypatch.setattr(manager.opt_log, 'plot_optimizer_trials', mock)

        if backend == 'processes':
            with run_warning:
                manager.start_manager()

            assert 't_stop' in manager.opt_log._storage[1]['metadata']
            assert manager.opt_log._storage[1]['metadata']['end_cond'] == "Forced GloMPO Termination"
            assert "Force terminated due to no feedback timeout." in manager.opt_log._storage[1]['messages']

    def test_too_long_hangingterm(self, manager, tmp_path, hanging_process, mask_psutil):
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                      bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector(HangOnEndOptimizer), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=2, backend='processes',
                      convergence_checker=MaxOptsStarted(3), killing_conditions=TrueHunter(2),
                      summary_files=2, force_terminations_after=30, split_printstreams=False)

        manager._optimizer_packs[1] = ProcessPackage(hanging_process, None, None, 1)
        manager.hunt_victims[1] = time() - 30
        manager._last_feedback[1] = time() - 30
        manager.opt_log.add_optimizer(1, None, time() - 60)

        with pytest.warns(RuntimeWarning, match="Forced termination signal sent to optimizer 1."):
            manager._inspect_children()

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_opt_error(self, backend, manager, tmp_path):
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector(ErrorOptimizer), working_dir=tmp_path,
                      overwrite_existing=True, max_jobs=1, backend=backend,
                      convergence_checker=MaxOptsStarted(2), summary_files=2, split_printstreams=False)

        with pytest.warns(RuntimeWarning, match="terminated in error"):
            manager.start_manager()

        assert 't_stop' in manager.opt_log._storage[1]['metadata']
        assert "Error termination (exitcode" in manager.opt_log._storage[1]['metadata']['end_cond']
        assert any(["Terminated in error with code" in message for message in manager.opt_log._storage[1]['messages']])

    @pytest.mark.parametrize("err", [RuntimeError, KeyboardInterrupt])
    def test_manager_error(self, err, manager, tmp_path, mask_psutil, monkeypatch):
        def mock_fill_slots(*args, **kwargs):
            raise err

        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5, bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector(SilentOptimizer), working_dir=tmp_path,
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
        else:  # 'processes_forced'
            opt_type = mp.Process
            opt_backend = 'processes'
            is_daemon = False

        manager = GloMPOManager()
        manager.setup(task=lambda x, y, z: x ** 2 + 3 * y ** 4 - z ** 0.5,
                      bounds=((0, 1), (0, 1), (0, 1)),
                      opt_selector=CycleSelector(SilentOptimizer), working_dir=tmp_path,
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

        def mock_save_traj(name, log_scale):
            gathered.append(log_scale)

        manager.setup(task=lambda x, y: x + y, bounds=((0, 1), (0, 1)),
                      opt_selector=CycleSelector(OptimizerTest1), working_dir=tmp_path, max_jobs=10,
                      summary_files=2, split_printstreams=False)

        manager.opt_log = BaseLogger(False)
        monkeypatch.setattr(manager.opt_log, "plot_trajectory", mock_save_traj)

        manager.o_counter = 1
        manager.opt_log.add_optimizer(1, OptimizerTest1.__name__, 0)
        for i, f in enumerate(fx):
            manager.opt_log.put_iteration(IterationResult(1, [0.5, 0.5], float(f), []))

        manager._save_log(Result([0.2, 0.3], 65.54, {}, {}), "GloMPO Convergence", "", tmp_path, 5)

        assert len(set(gathered)) == 1

    def test_status_message(self, manager):
        from glompo.core.manager import HAS_PSUTIL

        class FakeProcess:
            def is_alive(self):
                return True

        manager: GloMPOManager
        manager.t_start = time()
        manager.max_job = 10
        manager.opt_log = BaseLogger(False)
        manager.opt_log.add_optimizer(1, 'OptimizerTest1', '2020-10-30 20:30:13')
        for i in range(1, 10):
            manager.opt_log.put_iteration(IterationResult(1, [i], i, []))
        manager._optimizer_packs = {1: ProcessPackage(FakeProcess(), None, None, 1)}

        if HAS_PSUTIL:
            import psutil
            manager._process = psutil.Process()

        status = manager._build_status_message()
        assert all([header in status for header in ['Time Elapsed', 'Optimizers Alive', 'Slots Filled',
                                                    'Function Evaluations', 'Current Optimizer f_vals',
                                                    'Overall f_best:']])
        if HAS_PSUTIL:
            assert all([header in status for header in ['CPU Usage', 'Virtual Memory', 'System Load']])

    def test_resource_logging(self, manager, tmp_path):
        try:
            import psutil
        except (ImportError, ModuleNotFoundError):
            pytest.xfail("psutil not detected.")

        manager: GloMPOManager
        manager._process = psutil.Process()
        manager.t_start = time()
        summary = manager._summarise_resource_usage()

        assert summary == {'load_ave': [0], 'load_std': [0], 'mem_ave': '--', 'mem_max': '--',
                           'cpu_ave': 0, 'cpu_std': 0}

        for i in range(10):
            manager._build_status_message()

        summary = manager._summarise_resource_usage()

        assert len(summary['load_ave']) == 3
        assert len(summary['load_std']) == 3
        assert any([suffix in summary['mem_ave'] for suffix in ('B', 'kB', 'MB', 'GB')])
        assert any([suffix in summary['mem_max'] for suffix in ('B', 'kB', 'MB', 'GB')])
        assert summary['cpu_ave'] > 0
        assert summary['cpu_std'] > 0

    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_inf_spawn(self, manager, tmp_path, backend, caplog, capfd):
        class CrashingOptimizer(BaseOptimizer):
            def minimize(self, function: Callable[[Sequence[float]], float], x0: Sequence[float],
                         bounds: Sequence[Tuple[float, float]], callbacks: Callable = None, **kwargs) -> MinimizeResult:
                raise Exception

            def callstop(self, *args):
                pass

        manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                      bounds=[(0, 100)] * 3,
                      opt_selector=CycleSelector(CrashingOptimizer),
                      working_dir=tmp_path,
                      convergence_checker=MaxOptsStarted(10),
                      max_jobs=3,
                      backend=backend,
                      split_printstreams=False)
        with pytest.raises(RuntimeError, match="Optimizers spawning and crashing immediately."):
            while True:
                manager._fill_optimizer_slots()
                for i, pack in manager._optimizer_packs.items():
                    if pack.process.is_alive():
                        manager._check_signals(i)

    def test_stop_file(self, manager, tmp_path, monkeypatch, caplog):
        class ShutdownLogger:
            def __init__(self):
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args[0])

        caplog.set_level(logging.DEBUG, logger='glompo.manager')

        mock_shutdown_logger = ShutdownLogger()
        monkeypatch.setattr(manager, '_shutdown_job', mock_shutdown_logger)

        files = ['STOP_1', 'STOP_2', 'STOP_3', 'STOP_X']
        for file in files:
            (tmp_path / file).touch()
        manager.working_dir = tmp_path
        manager.graveyard = {1}
        manager._optimizer_packs = {2: ()}

        manager._is_manual_shutdowns()

        assert [(tmp_path / file).exists() for file in files] == [True, False, True, True]
        assert sorted([f"Error encountered trying to process STOP file '{tmp_path / 'STOP_X'}'",
                       f"Matching living optimizer not found for '{tmp_path / 'STOP_1'}'",
                       f"Matching living optimizer not found for '{tmp_path / 'STOP_3'}'",
                       "STOP file found for Optimizer 2"]) == sorted(caplog.messages)
        assert mock_shutdown_logger.calls == [2]

    @pytest.mark.skipif(not HAS_DILL, reason="dill package needed to test and use checkpointing.")
    def test_manual_checkpoint(self, manager, monkeypatch, tmp_path):
        class CheckpointCallLogger:
            def __init__(self):
                self.called = False

            def __call__(self):
                self.called = True

        mock_ck_logger = CheckpointCallLogger()
        monkeypatch.setattr(manager, 'checkpoint', mock_ck_logger)

        manager.working_dir = tmp_path
        (tmp_path / 'CHKPT').touch()
        manager._is_manual_checkpoints()
        assert not (tmp_path / 'CHKPT').exists()
        assert mock_ck_logger.called
        assert manager.checkpoint_control is None

    def test_write_summary(self, manager, tmp_path):
        manager.bounds = [Bound(0, 1)]
        manager.write_summary_file(tmp_path)
        assert (tmp_path / 'glompo_manager_log.yml').exists()

    @pytest.mark.mini
    @pytest.mark.parametrize('backend', ['processes', 'threads'])
    def test_mwe(self, backend, manager, save_outputs, tmp_path):
        class Task:
            def __call__(self, x):
                return self.f(x)

            def detailed_call(self, x):
                return self.f(x), self.df_dx(x), self.df_dy(x)

            def f(self, pt, delay=0.1):
                x, y = pt
                calc = -np.cos(0.2 * x)
                calc *= np.exp(-x ** 2 / 5000)
                calc /= 50 * np.sqrt(2 * np.pi)
                calc += 1e-6 * y ** 2
                sleep(delay)
                return calc

            def df_dx(self, pt):
                x, _ = pt
                calc = np.exp(-x ** 2 / 5000)
                calc *= x * np.cos(0.2 * x)
                calc /= 125000 * np.sqrt(2 * np.pi)
                calc += 0.00159577 * np.exp(-x ** 2 / 5000) * np.sin(0.2 * x)
                return calc

            def df_dy(self, pt):
                _, y = pt
                calc = 2e-6 * y
                return calc

        class SteepestGradient(BaseOptimizer):

            def __init__(self, max_iters, gamma, precision,
                         _opt_id=None,
                         _signal_pipe=None,
                         _results_queue=None,
                         _pause_flag=None,
                         workers=1,
                         backend='threads',
                         is_log_detailed=False):
                super().__init__(_opt_id, _signal_pipe, _results_queue, _pause_flag, workers, backend, is_log_detailed)
                self.max_iters = max_iters
                self.gamma = np.array(gamma)
                self.precision = precision
                self.terminate = False
                self.reason = None
                self.current_x = None
                self.result = MinimizeResult()

            def minimize(self, function: Task, x0: Sequence[float], bounds: Sequence[Tuple[float, float]],
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
                    fx, dx, dy = function.detailed_call(self.current_x)
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
                    self.message_manager(0, self.reason)
                print(f"Stopping due to {self.reason}")
                return self.result

            def callstop(self, *args):
                self.terminate = True
                self.reason = "manager termination"

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

        manager.setup(task=Task(), bounds=((-100, 100), (-100, 100)),
                      opt_selector=CycleSelector((SteepestGradient, {'max_iters': 10000,
                                                                     'precision': 1e-8,
                                                                     'gamma': [100, 100000]}, None)),
                      working_dir=tmp_path, overwrite_existing=True, max_jobs=3, backend=backend,
                      convergence_checker=KillsAfterConvergence(2, 1) | MaxFuncCalls(10000) | MaxSeconds(
                          session_max=60),
                      x0_generator=IntervalGenerator(), killing_conditions=MinFuncCalls(1000),
                      summary_files=3, visualisation=False, visualisation_args=None)
        result = manager.start_manager()
        assert np.all(np.isclose(result.x, 0, atol=1e-6))
        assert np.isclose(result.fx, -0.00797884560802864)
        assert result.origin['opt_id'] == 1
        assert result.origin['type'] == 'SteepestGradient'


@pytest.mark.skipif(not HAS_DILL, reason="dill package needed to test and use checkpointing.")
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
        (tmp_path / 'chkpt.tar.gz').touch()
        with pytest.warns(UserWarning, match="Overwriting existing checkpoint. To avoid this change the checkpoint"):
            manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                          bounds=[(0, 100)] * 3,
                          opt_selector=CycleSelector((RandomOptimizer, {'iters': 1000}, None)),
                          working_dir=tmp_path,
                          max_jobs=2,
                          summary_files=2,
                          backend='processes',
                          convergence_checker=None,
                          x0_generator=None,
                          killing_conditions=None,
                          checkpoint_control=CheckpointingControl(checkpoint_at_init=True,
                                                                  naming_format='chkpt',
                                                                  checkpointing_dir=tmp_path),
                          visualisation=True,
                          visualisation_args={'x_range': 2000})
        assert (tmp_path / 'chkpt.tar.gz').exists()
        assert not (tmp_path / 'chkpt').exists()
        with tarfile.open(tmp_path / 'chkpt.tar.gz', 'r:gz') as tfile:
            members = tfile.getnames()
            assert 'task' in members
            assert 'scope' in members
            assert 'manager' in members
            assert 'optimizers' in members
            assert all(['optimizers/' not in member for member in members])
        request.config.cache.set('init_checkpoint', str(tmp_path / 'chkpt.tar.gz'))

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
                      opt_selector=CycleSelector((RandomOptimizer, {'iters': 1000}, None)),
                      working_dir=tmp_path,
                      max_jobs=2,
                      backend=backend,
                      summary_files=1,
                      convergence_checker=MaxSeconds(session_max=3),
                      x0_generator=None,
                      killing_conditions=None,
                      checkpoint_control=CheckpointingControl(checkpoint_time_frequency=2.5,
                                                              naming_format='chkpt_%(count)',
                                                              checkpointing_dir=tmp_path),
                      visualisation=True,
                      visualisation_args={'record_movie': True, 'movie_kwargs': {'outfile': tmp_path / 'mv.mp4'}})
        with pytest.warns(RuntimeWarning, match="Movie saving is not supported"):
            manager.start_manager()

        assert manager.f_counter > 0
        assert manager.o_counter > 0
        assert manager.t_end - manager.t_start > 3
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
                                             checkpointing_control=None)

        init_f_count = manager.f_counter
        assert manager.task([0.2, 4, 6.1]) == 0.2 ** 2 + 3 * 4 ** 4 - 6.1 ** 0.5
        assert manager.working_dir == tmp_path
        assert manager.max_jobs == 2
        assert init_f_count > 0
        assert manager.o_counter == 2
        assert manager.visualisation
        assert 2.5 < manager.t_used < 3.5
        assert manager.t_start is None
        assert manager.opt_crashed is False
        assert manager.last_opt_spawn == (0, 0)

        manager.start_manager()
        assert (tmp_path / 'glompo_manager_log.yml').exists()
        assert manager.f_counter > init_f_count
        assert manager.t_end - manager.t_start < 3

    @pytest.mark.parametrize("delete, raises, warns",
                             [(['scope'], does_not_raise(), pytest.warns(RuntimeWarning, match="Could not load scope")),
                              (['manager'], does_not_raise(), pytest.raises(CheckpointingError,
                                                                            match="Error loading manager. Aborting.")),
                              (['optimizers/0001'], pytest.warns(RuntimeWarning,
                                                                 match="Failed to initialise opt"), does_not_raise()),
                              (['task'], does_not_raise(), pytest.raises(CheckpointingError,
                                                                         match="Failed to build task")),
                              (['optimizers/0001', 'optimizers/0002'], pytest.warns(RuntimeWarning,
                                                                                    match="Failed to initialise opt"),
                               pytest.raises(CheckpointingError, match="Unable to successfully built"))])
    def test_err_load(self, manager, tmp_path, delete, request, raises, warns):
        checkpoint_path = request.config.cache.get('mdpt_checkpoint', None)
        if not checkpoint_path or not Path(checkpoint_path).exists():
            pytest.xfail(reason="Checkpoint not found")

        shutil.copy(checkpoint_path, tmp_path)
        with tarfile.open(tmp_path / 'chkpt_000.tar.gz', 'r:gz') as tfile:
            tfile.extractall(tmp_path)
        for file in delete:
            (tmp_path / file).unlink()
            (tmp_path / file).touch()
        (tmp_path / 'chkpt_000.tar.gz').unlink()
        with tarfile.open(tmp_path / 'chkpt_000.tar.gz', 'x:gz') as tfile:
            tfile.add(tmp_path, recursive=True, arcname='')
        for file in tmp_path.iterdir():
            if file.name == 'chkpt_000.tar.gz':
                continue
            elif file.name == 'optimizers':
                shutil.rmtree(file, ignore_errors=True)
            else:
                file.unlink()

        with raises, warns:
            manager.load_checkpoint(tmp_path / 'chkpt_000.tar.gz')

    def test_conv_save(self, manager, tmp_path, mnkyptch_loggers, save_outputs, request):
        manager: GloMPOManager
        request.config.cache.set('conv_checkpoint', None)
        manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                      bounds=[(0, 100)] * 3,
                      opt_selector=CycleSelector((RandomOptimizer, {'iters': 100000}, None)),
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
                      summary_files=1)
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
        assert 3 < manager.t_used < 5
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

    @pytest.mark.parametrize('new_max_jobs, raises', [(1, pytest.raises(CheckpointingError, match="Insufficient max")),
                                                      (2, does_not_raise())])
    def test_new_maxjobs(self, input_files, manager, new_max_jobs, raises):
        with raises:
            with pytest.warns(UserWarning, match="The maximum number of jobs allowed is less than that demanded"):
                manager.load_checkpoint(input_files / 'mock_chkpt.tar.gz', max_jobs=new_max_jobs)
            assert manager._optimizer_packs[1].slots == 1
            assert manager._optimizer_packs[2].slots == 1

    @pytest.mark.parametrize('keep', [-1, 0, 1, 2, 3])
    def test_delete_other(self, tmp_path, manager, mnkyptch_loggers, keep):
        for i in range(4):
            (tmp_path / f'chkpt_00{i}.tar.gz').touch()

        manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                      bounds=[(0, 100)] * 3,
                      opt_selector=CycleSelector((RandomOptimizer, {'iters': 1000}, None)),
                      working_dir=tmp_path,
                      checkpoint_control=CheckpointingControl(checkpoint_at_init=True,
                                                              naming_format='chkpt_%(count)',
                                                              checkpointing_dir=tmp_path,
                                                              keep_past=keep))

        if keep == -1:
            assert all([(tmp_path / f'chkpt_00{i}.tar.gz').exists() for i in range(5)])
        else:
            survivors = lambda k: [False] * (4 - k) + [True] * (k + 1)
            assert [(tmp_path / f'chkpt_00{i}.tar.gz').exists() for i in range(5)] == survivors(keep)

    @pytest.mark.parametrize('raises_bool, raises', [(True, pytest.raises(CheckpointingError,
                                                                          match="Cannot pickle convergence_checker")),
                                                     (False, pytest.warns(UserWarning, match="Checkpointing failed"))])
    def test_chkpt_err(self, manager, tmp_path, raises_bool, raises):
        with raises:
            manager.setup(task=lambda x: x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5,
                          bounds=[(0, 100)] * 3,
                          opt_selector=CycleSelector((RandomOptimizer, {'iters': 1000}, None)),
                          working_dir=tmp_path,
                          checkpoint_control=CheckpointingControl(checkpoint_at_init=True,
                                                                  naming_format='chkpt_%(count)',
                                                                  checkpointing_dir=tmp_path,
                                                                  raise_checkpoint_fail=raises_bool))

    @pytest.mark.parametrize('f_count, o_count', [(50, 2), (50, 1)])
    def test_restart_log_truncate(self, f_count, o_count, manager, input_files, tmp_path):
        log_file = tmp_path / 'glompo_log.h5'

        shutil.copy(input_files / 'mock_log.h5', log_file)

        manager.f_counter = f_count
        manager.o_counter = o_count
        manager._is_restart = True
        manager.working_dir = tmp_path
        manager._checksum = 'correctchecksum'
        manager.opt_log = None

        with pytest.raises(AttributeError):
            with pytest.warns(RuntimeWarning, match="The log file (100 evaluations) has iterated past the checkpoint"):
                manager.start_manager()

        with tb.open_file(str(log_file.resolve()), 'r') as file:
            assert [n._v_name for n in file.iter_nodes('/')] == [f'optimizer_{i}' for i in range(1, o_count + 1)]
            for i in range(1, o_count + 1):
                table = file.get_node(f'/optimizer_{i}/iter_hist')
                assert max(table.col('call_id')) <= f_count

    def test_restart_log_checksum(self, manager, tmp_path, input_files):
        log_file = tmp_path / 'glompo_log.h5'

        shutil.copy(input_files / 'mock_log.h5', log_file)

        manager._is_restart = True
        manager.working_dir = tmp_path
        manager._checksum = 'WRONGchecksum'
        manager.opt_log = None

        with pytest.raises(KeyError, match="Checkpoint points to log file"):
            manager.start_manager()

    @pytest.mark.parametrize('backend', ['threads', 'processes'])
    def test_sync(self, backend, manager, tmp_path, caplog):
        """ Ensure checkpointing syncs optimizers correctly. The five optimizers used in this test represent the five
            conditions below:
                Optimizer 1 - Optimizing Normally.
                Optimizer 2 - Dead before checkpoint starts but in optimizer packs.
                Optimizer 3 - Pushed final iteration and messaged it at point is is captured for checkpoint.
                Optimizer 4 - Kill during results processing during checkpoint.
        """

        caplog.set_level(logging.DEBUG, logger='glompo')

        class Optimizer1(BaseOptimizer):
            """ Designed to be caught at a random point during an optimization run.
                (Must be accepted into checkpoint)
            """

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.stop = False

            def minimize(self, function: Callable[[Sequence[float]], float], x0: Sequence[float],
                         bounds: Sequence[Tuple[float, float]], callbacks: Callable = None, **kwargs) -> MinimizeResult:
                i = 0
                while not self.stop:
                    i += 1
                    if i > 10:
                        sleep(0.1)
                    function([0])
                    self.check_messages()
                    self.logger.debug("Checked messages")
                self.logger.debug("Exited manager loop")
                self.message_manager(0)
                self.logger.debug("Messaged termination")

            def callstop(self, *args):
                self.logger.debug("Callstop called.")
                self.stop = True

        class Optimizer2(BaseOptimizer):
            """ Process/thread already terminated when manager starts checkpoint but still in its optimizer_packs
                attribute.
                (Must NOT be accepted into checkpoint)
            """

            def minimize(self, function: Callable[[Sequence[float]], float], x0: Sequence[float],
                         bounds: Sequence[Tuple[float, float]], callbacks: Callable = None, **kwargs) -> MinimizeResult:
                self.logger.debug("Waiting to detect signal.")
                self._signal_pipe.poll(timeout=None)
                self.logger.debug("Signal detected, exiting.")

            def callstop(self, *args):
                pass

        class Optimizer3(BaseOptimizer):
            """ Pushed and messaged it final iteration but captured for a checkpoint before it was able to close.
                (Must NOT be accepted into checkpoint)
            """

            def minimize(self, function: Callable[[Sequence[float]], float], x0: Sequence[float],
                         bounds: Sequence[Tuple[float, float]], callbacks: Callable = None, **kwargs) -> MinimizeResult:
                self.message_manager(0)
                self.logger.debug("Messaged termination")
                self.logger.debug("Waiting on manager signal")
                self._signal_pipe.poll(timeout=None)
                self.logger.debug("Signal detected")
                self.check_messages()
                self.logger.debug("Exiting")

            def callstop(self, *args):
                pass

        class Optimizer4(Optimizer1):
            """ Killed by hunt during results processing within checkpoint.
                (Must NOT be accepted into checkpoint)
            """

        manager.setup(task=lambda x: sum(x),
                      bounds=[(0, 1)] * 10,
                      opt_selector=CycleSelector(Optimizer1, Optimizer2, Optimizer3, Optimizer4),
                      working_dir=tmp_path,
                      max_jobs=4,
                      backend=backend,
                      summary_files=2,
                      killing_conditions=TypeHunter(Optimizer4),
                      hunt_frequency=1,
                      checkpoint_control=CheckpointingControl(checkpointing_dir=tmp_path,
                                                              raise_checkpoint_fail=True,
                                                              naming_format='chkpt'),
                      split_printstreams=False)
        manager._fill_optimizer_slots()
        manager._checkpoint_optimizers(tmp_path)
        manager._toggle_optimizers(1)
        manager._stop_all_children()

        if backend == 'threads':  # pytest cannot capture logs from child processes
            for opt_id in range(1, 5):
                if opt_id == 2:
                    continue

                assert (f'glompo.optimizers.opt{opt_id}', 10, "Preparing for Checkpoint") \
                       in caplog.record_tuples

                assert (f'glompo.optimizers.opt{opt_id}', 10, "Wait signal messaged to manager, "
                                                              "waiting for reply...") in caplog.record_tuples

                assert (f'glompo.optimizers.opt{opt_id}', 10, "Instruction received. "
                                                              "Executing...") in caplog.record_tuples

                assert (f'glompo.optimizers.opt{opt_id}', 10, "Instructions processed. "
                                                              "Pausing until release...") in caplog.record_tuples

                assert (f'glompo.optimizers.opt{opt_id}', 10, "Pause released by manager. "
                                                              "Checkpointing completed.") in caplog.record_tuples

                if opt_id in {1, 6}:
                    assert (f'glompo.optimizers.opt{opt_id}', 10, "Executing: checkpoint_save") \
                           in caplog.record_tuples
                elif opt_id in {3, 4, 5}:
                    assert (f'glompo.optimizers.opt{opt_id}', 10, "Executing: _checkpoint_pass") \
                           in caplog.record_tuples

        assert [(tmp_path / f'optimizers/000{i}').exists() for i in range(1, 5)] == [True] + [False] * 3
