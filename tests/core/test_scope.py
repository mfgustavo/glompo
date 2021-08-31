from pathlib import Path
from time import sleep

import numpy as np
import pytest

pytest.importorskip('matplotlib.pyplot', reason="Matplotlib package needed to use these features.")

import matplotlib.pyplot as plt

from glompo.core.scope import GloMPOScope

plt.ion()


class TestScope:

    @pytest.fixture()
    def scope(self):
        scp = GloMPOScope()
        yield scp
        scp.close_fig()

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
    def test_init_keyerr(self, kwargs, tmp_path):
        with pytest.warns(UserWarning):
            scope = GloMPOScope(record_movie=True,
                                x_range=(0, 1000),
                                y_range=(0, 1000),
                                **kwargs)
            scope.setup_moviemaker(tmp_path / 'movie.mp4')
            scope._writer.cleanup()

    @pytest.mark.parametrize("max_val", [510, 910, 210, 80, 300, 310])
    def test_point_truncation(self, max_val, scope):
        scope.truncated = 300
        scope.add_stream(1)
        for i in range(0, max_val, 10):
            scope.update_optimizer(1, (i, i ** 2 / 6))
        scope._redraw_graph()

        x = scope.opt_streams[1].get_xdata()
        y = scope.opt_streams[1].get_ydata()

        if max_val > scope.truncated:
            assert min(x) == max_val - scope.truncated - 10
            assert min(y) == (max_val - scope.truncated - 10) ** 2 / 6
        else:
            assert min(x) == 0
            assert min(y) == 0
        assert max(x) == max_val - 10
        assert max(y) == (max_val - 10) ** 2 / 6

    @pytest.mark.parametrize("max_val", [0, 100, 200, 300])
    def test_deletion(self, max_val, scope):
        scope.truncated = 300
        scope.add_stream(1)
        scope.add_stream(2, "CustomOptimizer")
        for i in range(0, max_val, 10):
            scope.update_optimizer(1, (i, i ** 2 / 6))
        scope.update_optimizer(2, (600, 1))
        scope._redraw_graph(True)

        x = scope.opt_streams[1].get_xdata()
        y = scope.opt_streams[1].get_ydata()

        assert len(x) == 0
        assert len(y) == 0
        assert 1 in scope._dead_streams

        scope.update_optimizer(1, (max_val, max_val ** 2 / 6))
        assert 1 not in scope._dead_streams

    @pytest.mark.parametrize("path, log", [([1, 100, 100, 100], True), ([1, 100, 100, 100], False)])
    def test_log_scale(self, path, log, scope):
        scope.elitism = True
        scope.log_scale = log
        scope.add_stream(1)
        for x, y in enumerate(path):
            scope.update_optimizer(1, (x, y))

        y_vals = scope.opt_streams[1].get_ydata()

        assert all([y == int(not log) for y in y_vals])

    @pytest.mark.parametrize("record", [True, False])
    def test_checkpointing(self, record, tmp_path):
        pytest.importorskip('dill', reason="dill package needed to test and use checkpointing")

        scope = GloMPOScope(log_scale=False,
                            record_movie=record,
                            movie_kwargs={'outfile': Path(tmp_path, "test_gen_movie.mp4")})

        if record:
            scope.setup_moviemaker()
        scope.add_stream(1, None)
        scope.update_optimizer(1, (231, 789))
        scope.update_checkpoint(1)
        if record:
            warn = RuntimeWarning
            match = "Movie saving is not supported with checkpointing"
        else:
            warn = None
            match = ''
        with pytest.warns(warn, match=match):
            scope.checkpoint_save(tmp_path)
        assert Path(tmp_path, 'scope').exists()
        scope.close_fig()

        loaded_scope = GloMPOScope()
        loaded_scope.load_state(tmp_path)
        assert 1 in loaded_scope.opt_streams
        assert loaded_scope.opt_streams[1].get_xdata() == [231]
        assert loaded_scope.opt_streams[1].get_ydata() == [789]
        assert loaded_scope.gen_streams['chkpt'].get_xdata() == [231]
        assert loaded_scope.gen_streams['chkpt'].get_ydata() == [789]
        loaded_scope.close_fig()

    @pytest.mark.parametrize("record", [True, False])
    @pytest.mark.parametrize("setup", [True, False])
    def test_generate_movie(self, tmp_path, record, setup, save_outputs):
        scope = GloMPOScope(log_scale=True,
                            record_movie=record,
                            movie_kwargs={'outfile': Path(tmp_path, "test_gen_movie.mp4")})
        if setup:
            if record:
                scope.setup_moviemaker()
            else:
                with pytest.warns(UserWarning, match="Cannot initialise movie writer."):
                    scope.setup_moviemaker()

        scope.add_stream(1)
        scope.add_stream(2)

        if record and setup:
            for i in range(0, 510, 10):
                scope.update_optimizer(1, (i, np.sin(i) + 3))
                scope.update_optimizer(2, (i, np.cos(i) + 3))
            scope.update_kill(1)
            scope.update_pause(2)
            scope.update_checkpoint(2)
            scope.update_norm_terminate(2)

            scope.generate_movie()
            assert Path(tmp_path, "test_gen_movie.mp4").exists()
            sleep(0.5)  # Due to matplotlib semantics we need a pause here otherwise the stream will not close properly
        elif record and not setup:
            assert scope.record_movie
            assert not scope.is_setup
            with pytest.raises(RuntimeError, match="Cannot record movie without calling setup_moviemaker first."):
                scope.update_optimizer(1, (3, np.sin(3) + 3))
                scope._redraw_graph(True)
            with pytest.warns(RuntimeWarning, match="Exception caught while trying to save movie"):
                scope.generate_movie()
        elif not record:
            with pytest.warns(UserWarning, match="Unable to generate movie file as data was not collected"):
                scope.generate_movie()
