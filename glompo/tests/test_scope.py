import os

import numpy as np
import pytest

try:
    from glompo.core.scope import GloMPOScope

    import matplotlib

    matplotlib.use('qt5agg')

    import matplotlib.pyplot as plt

    plt.ion()
    if matplotlib.pyplot.isinteractive() and int(matplotlib.__version__.split('.')[0]) >= 3:
        HAS_MATPLOTLIB = True
    else:
        HAS_MATPLOTLIB = False
except (ModuleNotFoundError, ImportError):
    HAS_MATPLOTLIB = False


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Interactive-enabled matplotlib>=3.0 required to test the scope.")
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
    def test_init_keyerr(self, kwargs):
        with pytest.warns(UserWarning):
            scope = GloMPOScope(record_movie=True,
                                x_range=(0, 1000),
                                y_range=(0, 1000),
                                **kwargs)
            scope.setup_moviemaker()
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

    def test_checkpointing(self, scope, tmp_path):
        scope.add_stream(1, None)
        scope.update_optimizer(1, (1, 1))
        scope.update_checkpoint(1)
        scope.logger = None  # Pytest intercepts loggers making pickling impossible
        scope.checkpoint_save(tmp_path)
        assert 'scope' in os.listdir(tmp_path)

        loaded_scope = GloMPOScope()
        loaded_scope.load_state(tmp_path)
        assert 1 in loaded_scope.opt_streams
        assert loaded_scope.opt_streams[1].get_xdata() == [1]
        assert loaded_scope.opt_streams[1].get_ydata() == [1]
        assert loaded_scope.gen_streams['chkpt'].get_xdata() == [1]
        assert loaded_scope.gen_streams['chkpt'].get_ydata() == [1]
        loaded_scope.close_fig()

    def test_generate_movie(self, tmp_path, save_outputs):
        scope = GloMPOScope(log_scale=True,
                            record_movie=True,
                            movie_kwargs={'outfile': os.path.join(tmp_path, "test_gen_movie.mp4")})
        scope.setup_moviemaker()
        scope.add_stream(1)
        scope.add_stream(2)
        for i in range(0, 510, 10):
            scope.update_optimizer(1, (i, np.sin(i) + 3))
            scope.update_optimizer(2, (i, np.cos(i) + 3))
        scope.update_kill(1)
        scope.update_pause(2)
        scope.update_checkpoint(2)
        scope.update_norm_terminate(2)

        scope.generate_movie()

        assert os.path.exists(os.path.join(tmp_path, "test_gen_movie.mp4"))
