import inspect
import shutil
from pathlib import Path

import pytest

import glompo


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
def save_outputs(request, pytestconfig):
    """ Fixture which will save contents of a test's tmp_path to tests/_saved_outputs"""
    yield
    if pytestconfig.getoption('--save-outs'):
        dest_path = Path(__file__).parent / '_saved_outputs' / request.node.name
        src_path = request.getfixturevalue('tmp_path')
        shutil.rmtree(dest_path, ignore_errors=True)
        shutil.copytree(src_path, dest_path)


@pytest.fixture(scope='session')
def input_files():
    inputs_path = Path(inspect.getabsfile(glompo)).parent.parent / 'tests/_test_inputs'
    yield inputs_path
