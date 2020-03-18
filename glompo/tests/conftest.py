

import pytest


def pytest_addoption(parser):
    parser.addoption("--save-movie", action="store_true", help="does not delete the movie made by test_scope fro "
                                                              "visual inspection")
