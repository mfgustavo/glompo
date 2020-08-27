import os
import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption("--save-outs", "-S", action="store_true", help="does not delete outputs produced by the movie"
                                                                    "making test and (optional) minimization test for "
                                                                    "manual inspection.")
    parser.addoption("--run-minimize", "-M", action="store_true", help="runs a minimum working example of GloMPO "
                                                                       "minimisation")


def pytest_runtest_setup(item):
    if 'mini' in item.keywords and not item.config.getvalue("--run-minimize"):
        pytest.skip("need --run-minimize or -M option to run")


@pytest.fixture(scope="session", autouse=True)
def make_tmp(request):
    # Setup
    os.makedirs("_tmp", exist_ok=True)
    home_dir = os.getcwd()

    yield home_dir

    # Teardown
    os.chdir(home_dir)
    if request.config.getoption('-S'):
        os.makedirs("saved_test_outputs", exist_ok=True)
        for data in ('glomporecording.mp4', 'mini_test'):
            try:
                shutil.move("_tmp" + os.sep + data, "saved_test_outputs" + os.sep + data)
            except FileNotFoundError:
                pass

    shutil.rmtree('_tmp')


@pytest.fixture(scope="function", autouse=True)
@pytest.mark.usefixtures("make_tmp")
def move_to_tmp(make_tmp):
    os.chdir(make_tmp + os.sep + '_tmp')
