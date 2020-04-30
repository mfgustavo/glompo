

import pytest


def pytest_addoption(parser):
    parser.addoption("--save-outs", action="store_true", help="does not delete outputs produced by the movie making "
                                                              "test and (optional) minimization test for manual "
                                                              "inspection.")
    parser.addoption("--run-minimize", "-M", action="store_true", help="runs a minimum working example of GloMPO "
                                                                       "minimisation")
    parser.addoption("--run-scope-tests", "-S", action="store_true", help="runs test_scope.py. Off by default to "
                                                                          "avoid compatibility issues with matplotlib "
                                                                          "in some environments.")


def pytest_runtest_setup(item):
    if 'mini' in item.keywords and not item.config.getvalue("--run-minimize"):
        pytest.skip("need --run-minimize or -M option to run")
    if 'scope' in item.keywords and not item.config.getvalue('--run-scope-tests'):
        pytest.skip("need --run-scope-tests or -S option to run")
