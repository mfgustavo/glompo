import logging
from pathlib import Path

import numpy as np
import pytest
import tables as tb
import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except (ModuleNotFoundError, ImportError):
    from yaml import Dumper
    from yaml import Loader

from glompo.common.helpers import WorkInDirectory, LiteralWrapper, literal_presenter, nested_string_formatting, \
    unknown_object_presenter, generator_presenter, optimizer_selector_presenter, present_memory, FlowList, \
    flow_presenter, numpy_array_presenter, numpy_dtype_presenter, BoundGroup, bound_group_presenter, is_bounds_valid, \
    distance, glompo_colors, rolling_min, unravel, deepsizeof, infer_headers, SplitOptimizerLogs
from glompo.opt_selectors import BaseSelector, CycleSelector
from glompo.opt_selectors.spawncontrol import IterSpawnStop
from glompo.generators import RandomGenerator, BaseGenerator
from glompo.common.namedtuples import Bound
from glompo.optimizers.random import RandomOptimizer

yaml.add_representer(LiteralWrapper, literal_presenter, Dumper=Dumper)
yaml.add_representer(FlowList, flow_presenter, Dumper=Dumper)
yaml.add_representer(np.ndarray, numpy_array_presenter, Dumper=Dumper)
yaml.add_representer(BoundGroup, bound_group_presenter, Dumper=Dumper)
yaml.add_multi_representer(np.generic, numpy_dtype_presenter, Dumper=Dumper)
yaml.add_multi_representer(BaseSelector, optimizer_selector_presenter, Dumper=Dumper)
yaml.add_multi_representer(BaseGenerator, generator_presenter, Dumper=Dumper)
yaml.add_multi_representer(object, unknown_object_presenter, Dumper=Dumper)


class TestSplitLogging:

    def run_log(self, directory, propogate, formatter=logging.Formatter()):
        opt_filter = SplitOptimizerLogs(directory, propagate=propogate, formatter=formatter)
        opt_handler = logging.FileHandler(Path(directory, "propogate.txt"), "w")
        opt_handler.addFilter(opt_filter)

        opt_handler.setLevel('DEBUG')

        logging.getLogger("glompo.optimizers").addHandler(opt_handler)
        logging.getLogger("glompo.optimizers").setLevel('DEBUG')

        logging.getLogger("glompo.optimizers.opt1").debug('8452')
        logging.getLogger("glompo.optimizers.opt2").debug('9216')

    def test_split(self, tmp_path):
        self.run_log(tmp_path, False)
        with Path(tmp_path, "optimizer_1.log").open('r') as file:
            key = file.readline()
            assert key == '8452\n'

        with Path(tmp_path, "optimizer_2.log").open('r') as file:
            key = file.readline()
            assert key == '9216\n'

    def test_formatting(self, tmp_path):
        formatter = logging.Formatter("OPT :: %(message)s :: DONE")
        self.run_log(tmp_path, False, formatter)
        with Path(tmp_path, "optimizer_1.log").open('r') as file:
            key = file.readline()
            assert key == "OPT :: 8452 :: DONE\n"

        with Path(tmp_path, "optimizer_2.log").open('r') as file:
            key = file.readline()
            assert key == "OPT :: 9216 :: DONE\n"

    @pytest.mark.parametrize("propogate", [True, False])
    def test_propogate(self, propogate, tmp_path):
        self.run_log(tmp_path, propogate)
        with Path(tmp_path, "propogate.txt").open("r") as file:
            lines = file.readlines()

        if propogate:
            assert lines[0] == '8452\n'
            assert lines[1] == '9216\n'
            assert len(lines) == 2
        else:
            assert len(lines) == 0


@pytest.mark.parametrize('arr, output', [([0, 1, 2, 3, float('inf'), -1, 0, 1, -2, -4, -4],
                                          [0, 0, 0, 0, 0, -1, -1, -1, -2, -4, -4]),
                                         ([0, 1, 2, 3, float('nan'), -1, 0, 1, -4], [])])
def test_rolling_best(arr, output):
    if np.isnan(arr).any():
        with pytest.raises(AssertionError):
            rolling_min(arr)
    else:
        assert rolling_min(arr) == output


@pytest.mark.parametrize('obj, ret', [([0, [1, [2, [3, 4, 5], 6, [7, [8]]], 9], 10, [11, 12, 13],
                                        [14], 15, [[[[16]], '?']]], [*range(17), '?']),
                                      ('averylongstring', ['averylongstring'])])
def test_unravel(obj, ret):
    assert [*unravel(obj)] == ret


@pytest.mark.parametrize('obj, size', [([1, 2, 3, 4, [1, 2, 3, 4]], 200),
                                       (11231, 28),
                                       (1123131231, 32),
                                       ([121, {'ag': [1, 2, 3], 'b': [4, 5, 6]}, ('4', None), [3, 3, 3]],
                                        96 + 240 + 51 + 50 + 88 + 88 + 64 + 88)])
def test_deepsizeof(obj, size):
    assert deepsizeof(obj) == size


@pytest.mark.parametrize('obj, ret', [(123.4, {'result_0': tb.Float64Col(pos=0)}),
                                      (79, {'result_0': tb.Int64Col(pos=0)}),
                                      ("XXX", {'result_0': tb.StringCol(280, pos=0)}),
                                      (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                       {'result_0': tb.Col.from_dtype(np.dtype((int, (3, 3))), pos=0)}),
                                      ([[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                                       {'result_0': tb.Col.from_dtype(np.dtype((int, (3, 3))), pos=0)}),
                                      ((4.0, 5.0),
                                       {'result_0': tb.Col.from_dtype(np.dtype((float, (2,))), pos=0)}),
                                      (False, {'result_0': tb.BoolCol(pos=0)}),
                                      (None, {'result_0': tb.Float64Col(pos=0)}),
                                      (float('inf'), {'result_0': tb.Float64Col(pos=0)}),
                                      (float('nan'), {'result_0': tb.Float64Col(pos=0)}),
                                      (complex(1, 3), {'result_0': tb.ComplexCol(itemsize=16, pos=0)})])
def test_infer_headers(obj, ret):
    assert infer_headers([obj]) == ret


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
            is_bounds_valid(bnds)


def test_distance():
    assert distance([1] * 9, [-1] * 9) == 6


def test_work_in_directory(tmp_path):
    start_direc = Path.cwd()
    with WorkInDirectory(Path(tmp_path, 'a', 'b', 'c')):
        assert Path.cwd() == Path(tmp_path, 'a', 'b', 'c')
    assert Path.cwd().samefile(start_direc)


@pytest.mark.parametrize("opt_id", [10, 35, 53, 67, 73, 88, 200, None])
def test_colors(opt_id):
    plt = pytest.importorskip('matplotlib.pyplot', "Matplotlib package needed to use these features.")
    cols = pytest.importorskip('matplotlib.colors', "Matplotlib package needed to use these features.")
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


@pytest.mark.parametrize("mem_int, mem_str", [(123, '123B'),
                                              (1234, '1.2kB'),
                                              (1234567, '1.2MB'),
                                              (1234567890, '1.1GB'),
                                              (12345678901112, '11.2TB')])
def test_memory_presenter(mem_int, mem_str):
    assert present_memory(mem_int, 1) == mem_str


class MaxCallsCallback:
    def __init__(self, max_iter, calls_per_iter):
        self.max_iter = max_iter
        self.calls_per_iter = calls_per_iter
        self.iters_used = 0


@pytest.mark.parametrize("dump, load",
                         [(LiteralWrapper("This\n\tis\n\t\ta\n\t\t\tTest"),
                           '"This\\n\\tis\\n\\t\\ta\\n\\t\\t\\tTest"\n'),

                          (FlowList([2] * 3), "[2, 2, 2]\n"),

                          ([2] * 3, "- 2\n- 2\n- 2\n"),

                          (BoundGroup([Bound(0, 1)] * 5 + [Bound(3, 6)] * 4),
                           '(0, 1): [0, 1, 2, 3, 4]\n(3, 6): [5, 6, 7, 8]\n'),

                          (CycleSelector((RandomOptimizer, {'workers': 1, 'popsize': 10},
                                          {'callbacks': MaxCallsCallback(100, 1)}), allow_spawn=IterSpawnStop(300)),
                           'Selector: CycleSelector\nAllow Spawn:\n  IterSpawnStop:\n    max_calls: 300\n'
                           'Available Optimizers:\n  0:\n    type: RandomOptimizer\n    init_kwargs:\n'
                           '      workers: 1\n      popsize: 10\n    call_kwargs:\n      callbacks:\n        '
                           'MaxCallsCallback:\n          calls_per_iter: 1\n          iters_used: 0\n          '
                           'max_iter: 100\n'),

                          (RandomGenerator([(6, 7)] * 30), 'Generator: RandomGenerator\nn_params: 30\n'),

                          (np.full(5, 3), '[3, 3, 3, 3, 3]\n'),

                          (np.int64(4), '4\n'),

                          (np.float32(4), '4.0\n')])
def test_yaml_presenters(dump, load, tmp_path):
    with (tmp_path / 'dump.yml').open('w+') as file:
        yaml_dump = yaml.dump(dump, Dumper=Dumper, default_flow_style=False, sort_keys=False)
        print(repr(yaml_dump))
        assert yaml_dump == load
        file.write(yaml_dump)
        file.seek(0)
        yaml.load(file, Loader=Loader)
