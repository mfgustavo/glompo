import sys

import pytest

from glompo.core._backends import ChunkingQueue, CustomThread, ThreadPrintRedirect


def target(opt_id):
    print(f"Hello I am optimizer {opt_id}")
    print(f"{opt_id} closing")

    if opt_id == 1:
        raise RuntimeError(f"{opt_id} ran into a problem")


@pytest.fixture(scope='function')
def hack_sys_environment(tmp_path):
    (tmp_path / "glompo_optimizer_printstreams").mkdir()

    sys.stdout = ThreadPrintRedirect(sys.stdout)
    sys.stderr = ThreadPrintRedirect(sys.stderr)

    return


@pytest.fixture(scope='function')
def run_threads(tmp_path, request):
    threads = []
    for i in range(2):
        t = CustomThread(target=target,
                         args=(i,),
                         name=f'Opt{i}',
                         working_directory=tmp_path,
                         redirect_print=request.param)
        threads.append(t)
        t.start()
        t.join()

    assert threads[0].exitcode == 0
    assert threads[1].exitcode == -1

    return threads


@pytest.mark.parametrize('run_threads', [True], indirect=True)
def test_redirect(tmp_path, hack_sys_environment, run_threads):
    print_path = tmp_path / "glompo_optimizer_printstreams"

    for i in range(2):
        assert (print_path / f'printstream_{i:04}.out').exists()
        assert (print_path / f'printstream_{i:04}.out').read_text() == f"Hello I am optimizer {i}\n" \
                                                                       f"{i} closing\n" + ""

        assert (print_path / f'printstream_{i:04}.err').exists()
        if i == 1:
            assert f"{i} ran into a problem" in (print_path / f'printstream_{i:04}.err').read_text()


@pytest.mark.parametrize('run_threads', [False], indirect=True)
def test_no_redirect(tmp_path, capsys, run_threads):
    captured = capsys.readouterr()
    assert captured.out == "Hello I am optimizer 0\n" \
                           "0 closing\n" \
                           "Hello I am optimizer 1\n" \
                           "1 closing\n"
    assert "1 ran into a problem" in captured.err


def test_chunking_queue():
    q = ChunkingQueue(5, 10)

    for i in range(5):
        q.put_nowait(i)
        assert not q.has_cache()
        assert not q.fast_func

    q.put_nowait(5)
    assert q.fast_func
    assert q.cache == [5]

    for i in range(5):
        assert q.get_nowait() == [i]

    assert q.empty()

    for i in range(8):
        q.put_nowait(i)
        assert q.empty()

    q.put(100)
    assert q.get_nowait() == [5, 0, 1, 2, 3, 4, 5, 6, 7, 100]

    assert q.empty()

    q.put_nowait(True)
    q.put(False)
    assert q.get() == [True, False]
    assert q.empty()

    q.put(True)
    q.flush()
    assert q.get() == [True]
