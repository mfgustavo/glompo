

from functools import wraps
from typing import Callable, Sequence, Tuple
from time import time, sleep
from collections import namedtuple
import os
import shutil
import logging
import multiprocessing as mp

import pytest

from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from glompo.optimizers.cmawrapper import CMAOptimizer
from glompo.optimizers.gflswrapper import GFLSOptimizer
from glompo.common.namedtuples import IterationResult


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

    def save_state(self, *args):
        with open("savestate.txt", "w+") as file:
            file.write("Start\n")
            for i in dir(self):
                file.write(f"{i}\n")
            file.write("End")


class TestBase:

    package = namedtuple("package", "opti queue p_pipe event")

    @pytest.fixture()
    def mp_package(self):
        manager = mp.Manager()
        queue = manager.Queue()
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

    def test_savestate(self, mp_package):
        process = mp.Process(target=mp_package.opti.save_state, args=(None, None))
        process.start()
        process.join()

        with open("savestate.txt", "r") as file:
            lines = file.readlines()
            assert lines[0] == "Start\n"
            assert lines[-1] == "End"

        os.remove("savestate.txt")


class TestSubclassesGlompoCompatible:

    # Append new optimizer class names to this list to run tests for GloMPO compatibility
    available_classes = [CMAOptimizer, GFLSOptimizer]

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

        def __call__(self, *args, **kwrags):
            self.called += 1
            if self.called > self.max_iter:
                return "MaxIter"

            return None

    class Task:
        def __call__(self, x):
            return x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5

        def resids(self, pars, noise=True):
            return pars

    @pytest.fixture()
    def wrapped_task(self):
        def task_wrapper(func):
            @wraps(func)
            def wrapper(x):
                return func(x)
            return wrapper

        return task_wrapper(self.Task())

    @pytest.mark.parametrize("opti", available_classes)
    def test_result_in_queue(self, opti, mp_package, wrapped_task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.p_pipe,
                    pause_flag=mp_package.event)

        opti.minimize(function=wrapped_task,
                      x0=(0.5, 0.5, 0.5),
                      bounds=((0, 1), (0, 1), (0, 1)),
                      callbacks=self.MaxIter(10))

        assert not mp_package.queue.empty()
        assert isinstance(mp_package.queue.get_nowait(), IterationResult)

    @pytest.mark.parametrize("opti", available_classes)
    def test_final(self, opti, mp_package, wrapped_task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event)

        opti.minimize(function=wrapped_task,
                      x0=(0.5, 0.5, 0.5),
                      bounds=((0, 1), (0, 1), (0, 1)),
                      callbacks=self.MaxIter(10))

        res = False

        while not mp_package.queue.empty():
            res = mp_package.queue.get_nowait()

        # Make sure the final iteration is flagged as such
        assert res.final

        # Make sure a signal is sent that the optimizer is done
        assert mp_package.p_pipe.poll()
        assert mp_package.p_pipe.recv()[0] == 0

    @pytest.mark.parametrize("opti", available_classes)
    def test_callstop(self, opti, mp_package, wrapped_task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event)

        mp_package.p_pipe.send(1)
        opti.minimize(function=wrapped_task,
                      x0=(0.5, 0.5, 0.5),
                      bounds=((0, 1), (0, 1), (0, 1)),
                      callbacks=self.MaxIter(10))

        while not mp_package.queue.empty():
            assert mp_package.queue.get_nowait().n_iter < 10

    @pytest.mark.parametrize("opti", available_classes)
    def test_pause(self, opti, mp_package, wrapped_task):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event)

        p = mp.Process(target=opti.minimize,
                       kwargs={'function': wrapped_task,
                               'x0': (0.5, 0.5, 0.5),
                               'bounds': ((0, 1), (0, 1), (0, 1)),
                               'callbacks': self.MaxIter(10)})
        p.start()
        mp_package.event.clear()
        sleep(0.5)
        assert p.is_alive()

        mp_package.event.set()
        sleep(0.5)
        assert not p.is_alive()

    @pytest.mark.parametrize("opti", available_classes)
    def test_haslogger(self, opti, mp_package):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event)

        assert hasattr(opti, 'logger')
        assert isinstance(opti.logger, logging.Logger)

    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree("cmadata", ignore_errors=True)
            shutil.rmtree("tests/cmadata", ignore_errors=True)
        except FileNotFoundError:
            pass
