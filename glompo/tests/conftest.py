import os
import shutil
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--save-outs", action="store_true", help="does not delete outputs produced by the movie making "
                                                              "test and (optional) minimization test for manual "
                                                              "inspection.")
    parser.addoption("--run-minimize", "-M", action="store_true", help="runs a minimum working example of GloMPO "
                                                                       "minimisation")


def pytest_runtest_setup(item):
    if 'mini' in item.keywords and not item.config.getvalue("--run-minimize"):
        pytest.skip("need --run-minimize or -M option to run")


@pytest.fixture(scope="session", autouse=True)
def move_to_scratch(request):
    os.makedirs("_tmp", exist_ok=True)

    def fin():
        if '--save-outs' not in sys.argv:
            shutil.rmtree("_tmp", ignore_errors=True)

    request.addfinalizer(fin)
