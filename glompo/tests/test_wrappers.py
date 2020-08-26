import multiprocessing as mp
import os
import shutil

from glompo.common.wrappers import catch_user_interrupt, decorate_all_methods, process_print_redirect


def test_redirect():
    os.chdir("_tmp")
    os.makedirs("glompo_optimizer_printstreams", exist_ok=True)

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

    shutil.rmtree("glompo_optimizer_printstreams")
    os.chdir("..")


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
