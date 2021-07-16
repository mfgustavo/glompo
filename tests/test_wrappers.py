import multiprocessing as mp
from pathlib import Path

import pytest

from glompo.common.wrappers import catch_user_interrupt, decorate_all_methods, process_print_redirect, \
    needs_optional_package


def test_redirect(tmp_path):
    Path(tmp_path / "glompo_optimizer_printstreams").mkdir(parents=True, exist_ok=True)

    def func():
        print("redirect_test")
        raise RuntimeError("redirect_test_error")

    wrapped_func = process_print_redirect(1, tmp_path, func)
    p = mp.Process(target=wrapped_func)
    p.start()
    p.join()

    with Path(tmp_path, "glompo_optimizer_printstreams", "printstream_0001.out").open("r") as file:
        assert file.readline() == "redirect_test\n"

    with Path(tmp_path, "glompo_optimizer_printstreams", "printstream_0001.err").open("r") as file:
        assert any(["redirect_test_error" in line for line in file.readlines()])


def test_user_interrupt(capsys):
    def func():
        raise KeyboardInterrupt

    wrapped_func = catch_user_interrupt(func)
    wrapped_func()

    captured = capsys.readouterr()
    assert captured.out == "Interrupt signal received. Process stopping.\n"
    assert captured.err == ""


def test_decorate_all(capsys):
    def print_name(func):
        def wrapper(*args, **kwargs):
            print(func.__name__)
            return func(*args, **kwargs)

        return wrapper

    @decorate_all_methods(print_name)
    class Dummy:
        def dummy1(self):
            pass

        def dummy2(self):
            pass

        def dummy3(self):
            pass

    dummy = Dummy()
    dummy.dummy1()
    dummy.dummy2()
    dummy.dummy3()

    captured = capsys.readouterr()
    assert captured.out == "dummy1\ndummy2\ndummy3\n"
    assert captured.err == ""


@pytest.mark.parametrize('package, warns, ret', [('thispackagedefinitelydoesnotexist1048717812', ResourceWarning, None),
                                                 ('yaml', None, 765)])
def test_needs_package(package, warns, ret):
    @needs_optional_package(package)
    def wrapped_method():
        return 765

    with pytest.warns(warns, match="Unable to construct checkpoint without"):
        ans = wrapped_method()

    assert ans == ret

