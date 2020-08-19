import os
import shutil
import sys

import numpy as np
import pytest

has_matplotlib = False
try:
    from glompo.core.scope import GloMPOScope

    import matplotlib

    matplotlib.use('qt5agg')

    import matplotlib.pyplot as plt

    plt.ion()
    if matplotlib.pyplot.isinteractive() and int(matplotlib.__version__.split('.')[0]) >= 3:
        has_matplotlib = True
except (ModuleNotFoundError, ImportError):
    pass


@pytest.mark.skipif(not has_matplotlib, reason="Interactive-enabled matplotlib>=3.0 required to test the scope.")
class TestScope:

    @pytest.fixture()
    def scope(self):
        return GloMPOScope(x_range=None,
                           y_range=None,
                           events_per_flush=0,
                           record_movie=False,
                           interactive_mode=False)

    @pytest.mark.parametrize("kwargs", [{'x_range': -5},
                                        {'x_range': (500, 0)},
                                        {'y_range': (500, 0)}])
    def test_init_valerr(self, kwargs):
        with pytest.raises(ValueError):
            GloMPOScope(**kwargs)

    @pytest.mark.parametrize("kwargs", [{'x_range': 5.5},
                                        {'y_range': 5.5}])
    def test_init_typerr(self, kwargs):
        with pytest.raises(TypeError):
            GloMPOScope(**kwargs)

    @pytest.mark.parametrize("kwargs", [{'movie_kwargs': {'key1': 'xxx', 'key2': 'xxx'}},
                                        {'writer_kwargs': {'key': 'xxx'}}])
    def test_init_keyerr(self, kwargs):
        with pytest.warns(UserWarning):
            scope = GloMPOScope(record_movie=True,
                                x_range=(0, 1000),
                                y_range=(0, 1000),
                                **kwargs)
            scope._writer.cleanup()

    @pytest.mark.parametrize("i, palette", [(10, 1), (35, 2), (53, 3), (67, 4), (73, 5), (88, 6), (200, 7)])
    def test_colors(self, i, palette, scope):

        if i < 20:
            colors = plt.get_cmap("tab20")
            threshold = 0
            group = 1
        elif i < 40:
            colors = plt.get_cmap("tab20b")
            threshold = 20
            group = 2
        elif i < 60:
            colors = plt.get_cmap("tab20c")
            threshold = 40
            group = 3
        elif i < 69:
            colors = plt.get_cmap("Set1")
            threshold = 60
            group = 4
        elif i < 77:
            colors = plt.get_cmap("Set2")
            threshold = 69
            group = 5
        elif i < 89:
            colors = plt.get_cmap("Set3")
            threshold = 77
            group = 6
        else:
            colors = plt.get_cmap("Dark2")
            threshold = 89
            group = 7
        color = colors(i - threshold)

        scope.n_streams = i - 1
        scope.add_stream(0)
        assert color == scope.streams[0]['all_opt'].get_color()
        assert group == palette

    @pytest.mark.parametrize("max_val", [510, 910, 210, 80, 300, 310])
    def test_point_truncation(self, max_val, scope):
        scope.truncated = 300
        scope.add_stream(1)
        for i in range(0, max_val, 10):
            scope.update_optimizer(1, (i, i ** 2 / 6))
        scope._redraw_graph()

        x = scope.streams[1]['all_opt'].get_xdata()
        y = scope.streams[1]['all_opt'].get_ydata()

        if max_val > scope.truncated:
            assert min(x) == max_val - scope.truncated - 10
            assert min(y) == (max_val - scope.truncated - 10) ** 2 / 6
        else:
            assert min(x) == 0
            assert min(y) == 0
        assert max(x) == max_val - 10
        assert max(y) == (max_val - 10) ** 2 / 6

    @pytest.mark.filterwarnings("ignore:More than 20 figures")
    @pytest.mark.parametrize("max_val", [0, 100, 200, 300])
    def test_deletion(self, max_val, scope):
        scope.truncated = 300
        scope.add_stream(1)
        scope.add_stream(2)
        for i in range(0, max_val, 10):
            scope.update_optimizer(1, (i, i ** 2 / 6))
        scope.update_optimizer(2, (600, 1))
        scope._redraw_graph()

        x = scope.streams[1]['all_opt'].get_xdata()
        y = scope.streams[1]['all_opt'].get_ydata()

        assert len(x) == 0
        assert len(y) == 0
        assert 1 in scope._dead_streams

    @pytest.mark.filterwarnings("ignore:More than 20 figures")
    @pytest.mark.parametrize("path, log", [([1, 100, 100, 100], True), ([1, 100, 100, 100], False)])
    def test_deletion(self, path, log, scope):
        scope.elitism = True
        scope.log_scale = log
        scope.add_stream(1)
        for x, y in enumerate(path):
            scope.update_optimizer(1, (x, y))

        y_vals = scope.streams[1]['all_opt'].get_ydata()

        assert all([y == int(not log) for y in y_vals])

    @pytest.mark.filterwarnings("ignore:More than 20 figures")
    def test_generate_movie(self):
        scope = GloMPOScope(record_movie=True)
        scope.add_stream(1)
        scope.add_stream(2)
        for i in range(0, 510, 10):
            scope.update_optimizer(1, (i, np.sin(i)))
            scope.update_optimizer(2, (i, np.cos(i)))

        scope.generate_movie()

        assert os.path.exists("glomporecording.mp4")

    @classmethod
    def teardown_class(cls):
        try:
            if '--save-outs' not in sys.argv:
                os.remove("glomporecording.mp4")
            else:
                shutil.move("glomporecording.mp4", "_tmp/glompo_test_recording.mp4")
        except FileNotFoundError:
            pass
