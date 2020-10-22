import os
import tempfile

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


@pytest.fixture(scope='function')
def work_in_tmp():
    """ Some test require moving into a temporary directory. This creates and moves to a temporary path and returns
        at the end of the test.
    """
    home_dir = os.getcwd()

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        yield tmp_dir

    os.chdir(home_dir)
