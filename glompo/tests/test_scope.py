

import os
import shutil
import sys

import pytest
import numpy as np
import matplotlib.pyplot as plt

from glompo.core.scope import GloMPOScope


class TestScope:

    @pytest.fixture()
    def scope(self):
        return GloMPOScope(x_range=None,
                           y_range=None,
                           visualise_gpr=False,
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
                                visualise_gpr=True,
                                x_range=(0, 1000),
                                y_range=(0, 1000),
                                **kwargs)
            scope.writer.cleanup()

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
        color = colors(i-threshold)

        scope.n_streams = i-1
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
        assert 1 in scope.dead_streams

    @pytest.mark.filterwarnings("ignore:More than 20 figures")
    def test_generate_movie(self):
        scope = GloMPOScope(record_movie=True)
        scope.add_stream(1)
        scope.add_stream(2)
        for i in range(0, 510, 10):
            scope.update_optimizer(1, (i, np.sin(i)))
            scope.update_optimizer(2, (i, np.cos(i)))

            if i % 30:
                scope.t_last = 0
                scope.update_scatter(1, (i, np.sin(i)))
                scope.update_scatter(2, (i, np.cos(i)))
                scope.update_mean(1, 0, i/100)
                scope.update_mean(2, 1, i/100)
            if i == 50:
                scope.t_last = 0
                scope.update_opt_start(1)
            if i == 90:
                scope.t_last = 0
                scope.update_opt_end(1)
                scope.update_norm_terminate(1)
                scope.update_kill(2)

        scope.generate_movie()

        assert os.path.exists("glomporecording.mp4")

    @classmethod
    def teardown_class(cls):
        try:
            if '--save-outs' not in sys.argv:
                os.remove("glomporecording.mp4")
            else:
                shutil.move("glomporecording.mp4", "tests/outputs/glompo_test_recording.mp4")
        except FileNotFoundError:
            pass
