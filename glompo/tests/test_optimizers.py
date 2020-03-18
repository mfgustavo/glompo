

from typing import Callable, Sequence, Tuple
from time import time, sleep
from collections import namedtuple
import os
import pytest
import multiprocessing as mp

from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult


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
