import logging
import multiprocessing as mp
from collections import namedtuple
from pathlib import Path
from time import sleep, time
from typing import Callable, Sequence, Tuple, Union

import pytest

from glompo.common.namedtuples import IterationResult
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from glompo.optimizers.random import RandomOptimizer

# Append new optimizer classes to this list to run tests for GloMPO compatibility
# Expected: Tuple[Type[BaseOptimizer], Dict[str, Any] (init arguments), Dict[str, Any] (minimize arguments)]
available_classes = [(RandomOptimizer, {}, {})]
try:
    from glompo.optimizers.cmawrapper import CMAOptimizer

    available_classes.append((CMAOptimizer, {}, {'sigma0': 0.5}))
except ModuleNotFoundError:
    pass

try:
    from glompo.optimizers.gflswrapper import GFLSOptimizer

    available_classes.append((GFLSOptimizer, {}, {}))
except ModuleNotFoundError:
    pass

try:
    from glompo.optimizers.nevergrad import Nevergrad

    available_classes.append((Nevergrad, {}, {}))
except ModuleNotFoundError:
    pass


class PlainOptimizer(BaseOptimizer):
    needscaler = False

    def __init__(self, opt_id: int = None, signal_pipe: mp.connection.Connection = None, results_queue: mp.Queue = None,
                 pause_flag: mp.Event = None):
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag)
        self.terminate = False

    def minimize(self,
                 function: Callable,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        while not self.terminate:
            self.check_messages()

    def push_iter_result(self, *args):
        self._results_queue.put("item")

    def callstop(self, code=0):
        """
        Signal to terminate the :meth:`minimize` loop while still returning a result
        """
        self.terminate = True
        self.message_manager(code, MinimizeResult())

    def checkpoint_save(self, path: Union[Path, str]):
        with Path(path, "savestate.txt").open("w+") as file:
            file.write("Start\n")
            for i in dir(self):
                file.write(f"{i}\n")
            file.write("End")


class TestBase:
    package = namedtuple("package", "opti queue p_pipe event")

    @pytest.fixture(scope='function')
    def mp_package(self):
        manager = mp.Manager()
        queue = manager.Queue(10)
        p_pipe, c_pipe = mp.Pipe()
        event = manager.Event()
        opti = PlainOptimizer(1, c_pipe, queue, event)
        return self.package(opti, queue, p_pipe, event)

    def test_pipe(self, mp_package):
        process = mp.Process(target=mp_package.opti.minimize, args=(None, None, None))
        t_start = time()
        process.start()

        # Test sending
        sleep(0.01)
        mp_package.p_pipe.send((1, 888))
        process.join(timeout=1.5)
        assert time() - t_start < 2 and process.exitcode == 0

        # Test receiving
        mess = mp_package.p_pipe.recv()
        assert mess[0] == 888
        assert mess[1].success is False
        assert mess[1].x is None
        assert mess[1].fx == float('inf')

    def test_invalid_message(self, mp_package):
        process = mp.Process(target=mp_package.opti.minimize, args=(None, None, None))
        process.start()

        sleep(0.01)
        mp_package.p_pipe.send([8.0])
        mp_package.p_pipe.send(1)
        process.join(timeout=1.5)

    def test_push_result(self, mp_package, capfd):
        def minimize(self, *args, **kwargs) -> MinimizeResult:
            for i in range(1, 12):
                self.push_iter_result(self, IterationResult(1, i, 1, [i], i ** 2, False))
                print(i)

        mp_package.opti.minimize = minimize
        mp_package.opti.push_iter_result = BaseOptimizer.push_iter_result
        process = mp.Process(target=mp_package.opti.minimize, args=(mp_package.opti,))
        process.start()

        sleep(0.5)
        assert capfd.readouterr().out == "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n"
        sleep(1.5)
        assert capfd.readouterr().out == ""

        mp_package.queue.get_nowait()
        sleep(0.1)
        assert capfd.readouterr().out == "11\n"
        process.join()

    def test_event(self, mp_package):
        process = mp.Process(target=mp_package.opti.minimize, args=(None, None, None))
        t_start = time()
        process.start()

        sleep(0.1)
        mp_package.event.clear()
        mp_package.p_pipe.send(1)
        sleep(0.5)
        mp_package.event.set()
        process.join(timeout=1.5)

        assert process.exitcode == 0 and 0.5 < time() - t_start < 1.5

    def test_queue(self, mp_package):
        process = mp.Process(target=mp_package.opti.push_iter_result)
        process.start()
        process.join()
        assert mp_package.queue.get() == "item"

    def test_checkpointsave(self, mp_package, tmp_path):
        process = mp.Process(target=mp_package.opti.checkpoint_save, args=(tmp_path,))
        process.start()
        process.join()

        with Path(tmp_path, "savestate.txt").open("r") as file:
            lines = file.readlines()
            assert lines[0] == "Start\n"
            assert lines[-1] == "End"

    @pytest.mark.parametrize("exception", [KeyboardInterrupt, Exception])
    def test_min_wrapper(self, mp_package, exception, capfd):
        def minimize(*args, **kwargs) -> MinimizeResult:
            raise exception

        mp_package.opti.minimize = minimize

        process = mp.Process(target=mp_package.opti._minimize, args=(None, None, None))
        process.start()
        process.join()

        if exception is KeyboardInterrupt:
            assert "Interrupt signal received. Process stopping.\n" == capfd.readouterr().out
        else:
            assert process.exitcode != 0
            assert mp_package.p_pipe.recv()[0] == 9

    def test_prepare_checkpoint(self, mp_package, capfd):
        def checkpoint_save(*args, **kwargs):
            print("SAVED")

        # pytest does not support capturing logs from child processes
        def print_log(mess):
            print(mess)

        mp_package.opti._result_cache = IterationResult(1, 3, 1, [5], 700, False)
        mp_package.opti._FROM_MANAGER_SIGNAL_DICT[0] = checkpoint_save
        mp_package.opti.logger.debug = print_log
        mp_package.opti.logger.info = print_log
        mp_package.opti.logger.warning = print_log
        mp_package.opti.logger.error = print_log
        mp_package.opti.logger.critical = print_log

        process = mp.Process(target=mp_package.opti._prepare_checkpoint)
        process.start()

        sleep(0.3)
        captured = capfd.readouterr()
        assert captured.err == ''
        assert captured.out == "Preparing for Checkpoint\n" \
                               "Outstanding result found. Pushing to queue...\n" \
                               "Oustanding result (iter=3) pushed\n" \
                               "Wait signal messaged to manager, waiting for reply...\n"

        assert mp_package.queue.get_nowait() == IterationResult(1, 3, 1, [5], 700, False)
        assert mp_package.p_pipe.recv() == (1, None)

        mp_package.p_pipe.send((0, None))
        sleep(0.3)
        captured = capfd.readouterr()
        assert captured.out == "Instruction received. Executing...\n" \
                               "Received signal: (0, None)\n" \
                               "Executing: checkpoint_save\n" \
                               "SAVED\n" \
                               "Instructions processed. Pausing until release...\n"

        assert not mp_package.event.is_set()
        mp_package.event.set()
        sleep(0.3)
        captured = capfd.readouterr()
        assert captured.out == "Pause released by manager. Checkpointing completed.\n"
        process.join()


@pytest.mark.parametrize("opti, init_kwargs, call_kwargs", available_classes)
class TestSubclassesGlompoCompatible:
    package = namedtuple("package", "queue p_pipe c_pipe event")

    @pytest.fixture()
    def mp_package(self):
        manager = mp.Manager()
        queue = manager.Queue()
        p_pipe, c_pipe = mp.Pipe()
        event = manager.Event()
        event.set()
        return self.package(queue, p_pipe, c_pipe, event)

    class MaxIter:
        def __init__(self, max_iter: int):
            self.max_iter = max_iter
            self.called = 0

        def __call__(self, *args, **kwargs):
            self.called += 1
            if self.called > self.max_iter:
                return "MaxIter"

            return None

    @pytest.fixture
    def task(self):
        class Task:
            def __call__(self, x):
                return x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5

            def resids(self, pars, noise=True):
                return pars

        return Task()

    def test_result_in_queue(self, opti, init_kwargs, call_kwargs, mp_package, task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.p_pipe,
                    pause_flag=mp_package.event,
                    **init_kwargs)

        opti.minimize(function=task,
                      x0=(0.5, 0.5, 0.5),
                      bounds=((0, 1), (0, 1), (0, 1)),
                      callbacks=self.MaxIter(10),
                      **call_kwargs)

        assert not mp_package.queue.empty()
        assert isinstance(mp_package.queue.get_nowait(), IterationResult)

    def test_final(self, opti, init_kwargs, call_kwargs, mp_package, task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event,
                    **init_kwargs)

        opti.minimize(function=task,
                      x0=(0.5, 0.5, 0.5),
                      bounds=((0, 1), (0, 1), (0, 1)),
                      callbacks=self.MaxIter(10),
                      **call_kwargs)

        res = False

        i = 0
        while not mp_package.queue.empty():
            i += 1
            res = mp_package.queue.get_nowait()
            assert res.n_iter == i  # Ensure sequential and no double sending at end

        # Make sure the final iteration is flagged as such
        assert res.final

        # Make sure a signal is sent that the optimizer is done
        assert mp_package.p_pipe.poll()
        assert mp_package.p_pipe.recv()[0] == 0

    def test_callstop(self, opti, init_kwargs, call_kwargs, mp_package, task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event,
                    **init_kwargs)

        mp_package.p_pipe.send(1)
        opti.minimize(function=task,
                      x0=(0.5, 0.5, 0.5),
                      bounds=((0, 1), (0, 1), (0, 1)),
                      callbacks=self.MaxIter(10),
                      **call_kwargs)

        while not mp_package.queue.empty():
            assert mp_package.queue.get_nowait().n_iter < 10

    @pytest.mark.parametrize("task", [10], indirect=["task"])
    def test_pause(self, opti, init_kwargs, call_kwargs, mp_package, task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event,
                    **init_kwargs)

        p = mp.Process(target=opti.minimize,
                       kwargs={'function': task,
                               'x0': (0.5, 0.5, 0.5),
                               'bounds': ((0, 1), (0, 1), (0, 1)),
                               'callbacks': self.MaxIter(10),
                               **call_kwargs})
        p.start()
        mp_package.event.clear()
        p.join(1.5)
        assert p.is_alive()

        mp_package.event.set()
        p.join()
        assert not p.is_alive()

    def test_haslogger(self, opti, init_kwargs, call_kwargs, mp_package):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event,
                    **init_kwargs)

        assert hasattr(opti, 'logger')
        assert isinstance(opti.logger, logging.Logger)
        assert "glompo.optimizers.opt" in opti.logger.name
