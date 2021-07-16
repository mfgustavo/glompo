import datetime
from pathlib import Path

import pytest
import tables as tb

import glompo.core.optimizerlogger
from glompo.common.namedtuples import IterationResult
from glompo.core.optimizerlogger import BaseLogger, FileLogger

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except (ModuleNotFoundError, ImportError):
    HAS_MATPLOTLIB = False


@pytest.fixture(scope='class')
def filled_log(tmp_path_factory, request):
    log: BaseLogger = request.param(n_parms=1,
                                    expected_rows=90,
                                    build_traj_plot=True)

    log.open(tmp_path_factory.getbasetemp() / 'glompo_log.h5', 'w', 'correctchecksum')

    for i in range(1, 4):
        log.add_optimizer(i, 'Optimizer', datetime.datetime.now())
        log.add_iter_history(i, {'double': tb.Int32Col(),
                                 'triple': tb.Int32Col()})

    for i in range(30):
        y = i + 1
        log.put_iteration(IterationResult(1, [i], y, [2 * y, 3 * y]))
    log.put_metadata(1, "t_stop", datetime.datetime.now())
    log.put_metadata(1, "end_cond", "GloMPO Termination")

    for i in range(30):
        y = i + 11
        log.put_iteration(IterationResult(2, [i], y, [2 * y, 3 * y]))
    log.put_metadata(2, "t_stop", datetime.datetime.now())
    log.put_metadata(2, "end_cond", "Optimizer convergence")

    for i in range(30):
        y = i + 21
        log.put_iteration(IterationResult(3, [i], y, [2 * y, 3 * y]))
    log.put_metadata(3, "t_stop", datetime.datetime.now())
    log.put_metadata(3, "end_cond", "fmax condition met")

    log.put_message(2, "This is a test of the logger message system")

    return log


@pytest.mark.parametrize('filled_log', [BaseLogger, FileLogger], indirect=True)
class TestLogger:
    @pytest.mark.parametrize('opt_id', [*range(1, 4), None])
    def test_get_best(self, opt_id, filled_log):
        assert filled_log.get_best_iter(opt_id)['fx'] == (1 + 10 * (opt_id - 1) if opt_id is not None else 1)

    @pytest.mark.parametrize('opt_id', range(1, 4))
    def test_history(self, filled_log, opt_id):
        x = filled_log.get_history(opt_id, 'x')
        c = filled_log.get_history(opt_id, 'call_id')
        f = filled_log.get_history(opt_id, 'fx')

        assert x == [[i] for i in range(30)]
        assert f == [i + 10 * (opt_id - 1) for i in range(1, 31)]
        assert c == [*range(1 + 30 * (opt_id - 1), 31 + 30 * (opt_id - 1))]

    def test_message(self, filled_log):
        assert filled_log._storage[2]['messages'] == ["This is a test of the logger message system"]

    def test_contains(self, filled_log):
        assert all([i in filled_log for i in range(1, 4)])

    def test_opts(self, filled_log):
        assert filled_log.n_optimizers == 3

    def test_len(self, filled_log):
        assert len(filled_log) == 90

    @pytest.mark.parametrize('opt_id', range(1, 4))
    def test_opt_len(self, filled_log, opt_id):
        assert filled_log.len(opt_id) == 30

    def test_metadata(self, filled_log):
        assert filled_log.get_metadata(1, "end_cond") == "GloMPO Termination"
        assert filled_log.get_metadata(2, "end_cond") == "Optimizer convergence"
        assert filled_log.get_metadata(3, "end_cond") == "fmax condition met"

    def test_checkpoint_save(self, request, filled_log, tmp_path):
        filled_log.checkpoint_save(tmp_path)
        assert (tmp_path / 'opt_log').exists()
        request.config.cache.set('log_checkpoint', str(tmp_path / 'opt_log'))

    def test_checkpoint_load(self, filled_log, request):
        load_path = request.config.cache.get('log_checkpoint', None)
        if not load_path or not Path(load_path).exists():
            pytest.xfail('Successfully saved checkpoint not found.')

        log = BaseLogger.checkpoint_load(load_path)

        assert all([i in log for i in range(1, 4)])
        assert all([log.len(i) == 30 for i in range(1, 4)])
        assert log.largest_eval == 50
        assert log.get_best_iter() == {'opt_id': 1, 'x': [0], 'fx': 1, 'type': 'Optimizer', 'call_id': 1}

    @pytest.mark.skipif(not HAS_MATPLOTLIB,
                        reason="Matplotlib package needed to use these features.")
    @pytest.mark.parametrize("log_scale", [True, False])
    @pytest.mark.parametrize("best_fx", [True, False])
    def test_plot_traj(self, filled_log, log_scale, best_fx, tmp_path, save_outputs):
        filled_log.plot_trajectory(Path(tmp_path, 'traj.png'), log_scale, best_fx)
        assert Path(tmp_path, 'traj.png').exists()

    @pytest.mark.skipif(not HAS_MATPLOTLIB,
                        reason="Matplotlib package needed to use these features.")
    def test_plot_opts(self, tmp_path, filled_log, save_outputs):
        filled_log.plot_optimizer_trials(tmp_path)
        for i in range(1, 4):
            assert Path(tmp_path, f"opt{i}_parms.png").exists()

    @pytest.mark.parametrize('build_traj_plot', [True, False])
    def test_clear_cache(self, filled_log, build_traj_plot):
        filled_log.build_traj_plot = build_traj_plot
        filled_log.clear_cache(2)
        if build_traj_plot:
            assert [*filled_log._storage.keys()] == [1, 2, 3]
        else:
            assert [*filled_log._storage.keys()] == [1, 3]

    def test_file(self, filled_log, tmp_path_factory):
        if not isinstance(filled_log, FileLogger):
            pytest.skip("No file created by BaseLogger")

        filled_log.flush()
        filled_log.close()

        with tb.open_file(tmp_path_factory.getbasetemp() / 'glompo_log.h5') as file:
            assert [n._v_name for n in file.iter_nodes('/')] == [f'optimizer_{i}' for i in range(1, 4)]
            for i in range(1, 4):
                table = file.get_node(f'/optimizer_{i}/iter_hist')
                assert len(table.col('call_id')) == 30
