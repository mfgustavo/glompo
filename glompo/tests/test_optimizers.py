import logging
import multiprocessing as mp
import pickle
from pathlib import Path
from time import sleep, time
from typing import Callable, NamedTuple, Sequence, Tuple

import dill
import pytest

from glompo.common.namedtuples import IterationResult
from glompo.optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from glompo.optimizers.random import RandomOptimizer

# Append new optimizer classes to this list to run tests for GloMPO compatibility
# Expected: Tuple[Type[BaseOptimizer], Dict[str, Any] (init arguments), Dict[str, Any] (minimize arguments)]
AVAILABLE_CLASSES = {'RandomOptimizer': (RandomOptimizer, {}, {})}
try:
    from glompo.optimizers.cmawrapper import CMAOptimizer

    AVAILABLE_CLASSES['CMAOptimizer'] = (CMAOptimizer, {}, {'sigma0': 0.5})
except ModuleNotFoundError:
    pass

try:
    from glompo.optimizers.gflswrapper import GFLSOptimizer

    AVAILABLE_CLASSES['GFLSOptimizer'] = (GFLSOptimizer, {}, {})
except ModuleNotFoundError:
    pass

try:
    from glompo.optimizers.nevergrad import Nevergrad

    AVAILABLE_CLASSES['Nevergrad'] = (Nevergrad, {}, {})
except ModuleNotFoundError:
    pass


class PlainOptimizer(BaseOptimizer):

    def __init__(self, opt_id: int = None, signal_pipe: mp.connection.Connection = None, results_queue: mp.Queue = None,
                 pause_flag: mp.Event = None):
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag)
        self.terminate = False
        self.i = 0

    def minimize(self,
                 function: Callable,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs):
        while not self.terminate:
            self.i += 1
            self.check_messages()

    def push_iter_result(self, *args):
        self._results_queue.put("item")

    def callstop(self, code=0):
        """
        Signal to terminate the :meth:`minimize` loop while still returning a result
        """
        self.terminate = True
        self.message_manager(code, MinimizeResult())


class MPPackage(NamedTuple):
    queue: mp.Queue
    p_pipe: mp.Pipe
    c_pipe: mp.Pipe
    event: mp.Event


@pytest.fixture(scope='function')
def mp_package():
    manager = mp.Manager()
    queue = manager.Queue(10)
    p_pipe, c_pipe = mp.Pipe()
    event = manager.Event()
    event.set()
    return MPPackage(queue, p_pipe, c_pipe, event)


class MaxIter:
    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.called = 0

    def __call__(self, *args, **kwargs):
        self.called += 1
        if self.called >= self.max_iter:
            return "MaxIter"

        return None


class TestBase:

    @pytest.fixture(scope='function')
    def intercept_logging(self):
        """ pytest does not support capturing logs from child processes. This fixture intercepts and turns them into
            print statements.
        """

        def print_log(mess):
            print(mess)

        logger = logging.getLogger('glompo.optimizers.opt1')
        orig_methods = (logger.debug, logger.info, logger.warning, logger.error, logger.critical)

        logger.debug = print_log
        logger.info = print_log
        logger.warning = print_log
        logger.error = print_log
        logger.critical = print_log

        yield

        logger.debug = orig_methods[0]
        logger.info = orig_methods[1]
        logger.warning = orig_methods[2]
        logger.error = orig_methods[3]
        logger.critical = orig_methods[4]

    def test_pipe(self, mp_package):
        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        process = mp.Process(target=opti.minimize, args=(None, None, None))
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
        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        process = mp.Process(target=opti.minimize, args=(None, None, None))
        process.start()

        sleep(0.01)
        mp_package.p_pipe.send([8.0])
        mp_package.p_pipe.send(1)
        process.join(timeout=1.5)

    def test_push_result(self, mp_package, capfd):
        def minimize(self, *args, **kwargs):
            for i in range(1, 12):
                self.push_iter_result(self, IterationResult(1, i, 1, [i], i ** 2, False))
                print(i)

        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        opti.minimize = minimize
        opti.push_iter_result = BaseOptimizer.push_iter_result
        process = mp.Process(target=opti.minimize, args=(opti,))
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
        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        process = mp.Process(target=opti.minimize, args=(None, None, None))
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
        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        process = mp.Process(target=opti.push_iter_result)
        process.start()
        process.join()
        assert mp_package.queue.get() == "item"

    def test_checkpointsave(self, mp_package, tmp_path, capfd):
        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        opti.workers = 685
        process = mp.Process(target=opti.minimize, args=(None, None, None))
        process.start()

        mp_package.p_pipe.send((0, tmp_path / '0001'))
        mp_package.p_pipe.send(1)
        process.join()

        assert not capfd.readouterr().err
        with (tmp_path / "0001").open("rb") as file:
            data = dill.load(file)

            assert len(data) == 3
            assert data['terminate'] is False
            assert data['i'] > 0
            assert data['workers'] == 685

            loaded_opt = PlainOptimizer.checkpoint_load(tmp_path / '0001', 1)
            assert loaded_opt.i == data['i']
            assert loaded_opt.terminate is False
            assert loaded_opt.workers == 685

    @pytest.mark.parametrize("exception", [KeyboardInterrupt, Exception])
    def test_min_wrapper(self, mp_package, exception, capfd):
        def minimize(*args, **kwargs) -> MinimizeResult:
            raise exception

        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        opti.minimize = minimize

        process = mp.Process(target=opti._minimize, args=(None, None, None))
        process.start()
        process.join()

        if exception is KeyboardInterrupt:
            assert "Interrupt signal received. Process stopping.\n" == capfd.readouterr().out
        else:
            assert process.exitcode != 0
            assert mp_package.p_pipe.recv()[0] == 9

    def test_prepare_checkpoint(self, mp_package, capfd, intercept_logging):
        def checkpoint_save(*args, **kwargs):
            print("SAVED")

        opti = PlainOptimizer(1, mp_package.c_pipe, mp_package.queue, mp_package.event)
        opti._result_cache = IterationResult(1, 3, 1, [5], 700, False)
        opti._FROM_MANAGER_SIGNAL_DICT[0] = checkpoint_save

        process = mp.Process(target=opti._prepare_checkpoint, daemon=True)
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


@pytest.mark.parametrize("opti, init_kwargs, call_kwargs", AVAILABLE_CLASSES.values())
class TestSubclassesGlompoCompatible:

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
                      callbacks=MaxIter(10),
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
                      callbacks=MaxIter(10),
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
                      callbacks=MaxIter(10),
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
                               'callbacks': MaxIter(10),
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

    def test_checkpointing(self, opti, init_kwargs, call_kwargs, mp_package, task, tmp_path):
        opti = opti(results_queue=mp_package.queue,
                    signal_pipe=mp_package.c_pipe,
                    pause_flag=mp_package.event,
                    **init_kwargs)

        opti.added_to_check = 555
        assert not opti.is_restart

        opti.minimize(function=task,
                      x0=(0.5, 0.5, 0.5),
                      bounds=((0, 1), (0, 1), (0, 1)),
                      callbacks=MaxIter(5),
                      **call_kwargs)
        opti.checkpoint_save(tmp_path / '0001')

        iter_count = 0
        while not mp_package.queue.empty():
            iter_count += 1
            assert mp_package.queue.get_nowait().n_iter == iter_count

        loaded_opti = opti.checkpoint_load(tmp_path / '0001',
                                           results_queue=mp_package.queue,
                                           signal_pipe=mp_package.c_pipe,
                                           pause_flag=mp_package.event,
                                           **init_kwargs)

        assert loaded_opti.added_to_check == 555
        assert loaded_opti.is_restart

        loaded_opti.minimize(function=task,
                             x0=(0.5, 0.5, 0.5),
                             bounds=((0, 1), (0, 1), (0, 1)),
                             callbacks=MaxIter(5),
                             **call_kwargs)

        while not mp_package.queue.empty():
            iter_count += 1
            assert mp_package.queue.get_nowait().n_iter == iter_count


@pytest.mark.skipif('CMAOptimizer' not in AVAILABLE_CLASSES, reason="CMA not loaded.")
class TestCMA:
    """ Specific CMAOptimizer tests not covered by TestSubclassesGlompoCompatible """

    class FakeES:
        callbackstop = 0

    @pytest.fixture()
    def optimizer(self, mp_package):
        return CMAOptimizer(opt_id=1,
                            signal_pipe=mp_package.c_pipe,
                            results_queue=mp_package.queue,
                            pause_flag=mp_package.event,
                            workers=1,
                            backend='threads',
                            popsize=3)

    @pytest.fixture()
    def task(self):
        def f(x):
            sleep(x[0])
            return x[0] ** 2 + 3 * x[1] ** 4 - x[2] ** 0.5

        return f

    @pytest.mark.parametrize('workers', [1, 3])
    @pytest.mark.parametrize('backend', ['threads', 'processes'])
    def test_parallel_map(self, backend, optimizer, workers, task, caplog):
        if workers == 3 and backend == 'processes':
            pytest.xfail("Known pytest bug cannot test ProcessPoolExecutor in Pytest environment.")

        caplog.set_level(logging.DEBUG, logger="glompo.optimizers.opt1")
        optimizer.workers = workers
        optimizer._backend = backend
        optimizer.es = self.FakeES()

        t_start = time()
        fx = optimizer._parallel_map(task, [[0.5] * 3] * 3)
        t_end = time()

        assert len(fx) == 3
        assert len(set(fx)) == 1
        if workers == 1:
            assert 1.5 < t_end - t_start < 2
            assert caplog.messages == ["Executing serially"]
        else:
            pool_executor = "ThreadPoolExecutor" if backend == 'threads' else "ProcessPoolExecutor"
            assert 0.5 < t_end - t_start < 1
            assert caplog.messages == [f"Executing within {pool_executor} with 3 workers",
                                       "Result 1/3 returned.", "Result 2/3 returned.", "Result 3/3 returned."]

    def test_interrupt_calc(self, optimizer, task, caplog, mp_package):
        assert mp_package.queue is optimizer._results_queue
        optimizer.workers = 2
        optimizer._backend = 'threads'
        optimizer.es = self.FakeES()
        optimizer.result = MinimizeResult()
        caplog.set_level(logging.DEBUG, logger="glompo.optimizers.opt1")

        mp_package.p_pipe.send((1, None))
        fx = optimizer._parallel_map(task, [[0] * 3] + [[0.5] * 3] * 9)
        assert len(fx) == 3
        assert caplog.messages == ["Executing within ThreadPoolExecutor with 2 workers",
                                   "Result 1/10 returned.",
                                   "Received signal: (1, None)",
                                   "Executing: callstop",
                                   "Calling stop. Reason = None",
                                   "Stop command received during function evaluations.",
                                   "Aborted 7 calls."]

    def test_save(self, monkeypatch, tmp_path, optimizer):
        monkeypatch.chdir(tmp_path)
        optimizer.keep_files = True
        optimizer.minimize(function=lambda x: x[0] ** 2 + x[1] ** 2,
                           x0=[1, 1],
                           bounds=[(-3, 3), (-3, 3)],
                           sigma0=1,
                           callbacks=MaxIter(10))
        dump = Path('cma_opt1_results.pkl')
        assert dump.exists()
        with dump.open('rb') as file:
            pickle.load(file)
