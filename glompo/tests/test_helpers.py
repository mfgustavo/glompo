import os
from os.path import join as pjoin

import pytest

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as cols

    HAS_MATPLOTLIB = True
except (ModuleNotFoundError, ImportError):
    HAS_MATPLOTLIB = False

from glompo.common.helpers import FileNameHandler, distance, is_bounds_valid, nested_string_formatting, glompo_colors


def test_string():
    assert nested_string_formatting("[TrueHunter() AND\n"
                                    "[[TrueHunter() OR\n"
                                    "[FalseHunter() AND\n"
                                    "[TrueHunter() OR\n"
                                    "FalseHunter()]]]\n"
                                    "OR\n"
                                    "FalseHunter()]]") == \
           "TrueHunter() AND\n" \
           "[\n" \
           " [\n" \
           "  TrueHunter() OR\n" \
           "  [\n" \
           "   FalseHunter() AND\n" \
           "   [\n" \
           "    TrueHunter() OR\n" \
           "    FalseHunter()\n" \
           "   ]\n" \
           "  ]\n" \
           " ]\n" \
           " OR\n" \
           " FalseHunter()\n" \
           "]"


def test_string_with_result():
    assert nested_string_formatting("[TrueHunter() = None AND\n"
                                    "[[TrueHunter() = None OR\n"
                                    "[FalseHunter() = None AND\n"
                                    "[TrueHunter() = None OR\n"
                                    "FalseHunter() = None]]]\n"
                                    "OR\n"
                                    "FalseHunter() = None]]") == \
           "TrueHunter() = None AND\n" \
           "[\n" \
           " [\n" \
           "  TrueHunter() = None OR\n" \
           "  [\n" \
           "   FalseHunter() = None AND\n" \
           "   [\n" \
           "    TrueHunter() = None OR\n" \
           "    FalseHunter() = None\n" \
           "   ]\n" \
           "  ]\n" \
           " ]\n" \
           " OR\n" \
           " FalseHunter() = None\n" \
           "]"


@pytest.mark.parametrize('bnds, output', [([(0, 1)] * 5, True),
                                          ([(1, -1)] * 5, False),
                                          ([(0, float('inf'))] * 5, False)])
def test_bounds(bnds, output):
    assert is_bounds_valid(bnds, raise_invalid=False) == output
    if not output:
        with pytest.raises(ValueError):
            is_bounds_valid(bnds, raise_invalid=True)


def test_distance():
    assert distance([1] * 9, [-1] * 9) == 6


def test_file_name_handler():
    start_direc = os.getcwd()
    with FileNameHandler(pjoin("test_helpers", "fnh")) as name:
        assert os.getcwd() == pjoin(start_direc, 'test_helpers')
        assert name == 'fnh'
    assert os.getcwd() == start_direc


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Requires matplotlib to use this function.")
@pytest.mark.parametrize("opt_id", [10, 35, 53, 67, 73, 88, 200, None])
def test_colors(opt_id):
    if opt_id:
        if opt_id < 20:
            colors = plt.get_cmap("tab20")
            threshold = 0
        elif opt_id < 40:
            colors = plt.get_cmap("tab20b")
            threshold = 20
        elif opt_id < 60:
            colors = plt.get_cmap("tab20c")
            threshold = 40
        elif opt_id < 69:
            colors = plt.get_cmap("Set1")
            threshold = 60
        elif opt_id < 77:
            colors = plt.get_cmap("Set2")
            threshold = 69
        elif opt_id < 89:
            colors = plt.get_cmap("Set3")
            threshold = 77
        else:
            colors = plt.get_cmap("Dark2")
            threshold = 89
        color = colors(opt_id - threshold)
        assert color == glompo_colors(opt_id)
    else:
        assert isinstance(glompo_colors(), cols.ListedColormap)
