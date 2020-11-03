import inspect
import os
import pickle
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

try:
    from scm.params.core.opt_components import _Step, _LossEvaluator, EvaluatorReturn
    from scm.params.common.parallellevels import ParallelLevels
    from scm.params.common.reaxff_converter import geo_to_params, trainset_to_params
    from scm.params.core.dataset import DataSet, Loss, SSE
    from scm.params.core.jobcollection import JobCollection
    from scm.params.core.opt_components import LinearParameterScaler, _Step
    from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
    from scm.params.parameterinterfaces.reaxff import ReaxParams
    from scm.plams.core.errors import ResultsError
    from scm.plams.interfaces.adfsuite.reaxff import reaxff_control_to_settings

    HAS_PARAMS = True
except (ModuleNotFoundError, ImportError):
    HAS_PARAMS = False

from glompo.interfaces.params import _FunctionWrapper, ReaxFFError


class FakeLossEvaluator(_LossEvaluator):
    def check(self):
        pass

    def __call__(self, x):
        return EvaluatorReturn(100, x, self.name, self.ncalled, self.interface,
                               None, [5, 2, -1, -9], [0.10, 0.25, 0.30, 0.35], 0)


@pytest.mark.skipif(not HAS_PARAMS, reason="SCM ParAMS needed to test and use the ParAMS interface.")
class TestParamsStep:
    """ This class of tests ensures that the scm.params._Step instance given to the GloMPOManager has the attributes
        expected.
    """

    @pytest.fixture()
    def params_func(self):
        loss_eval_list = []
        for i in range(3):
            loss_eval = FakeLossEvaluator(f'foo{i}', None, None, None, None,
                                          None, None, None, None, False, False, True, None)
            loss_eval_list.append(loss_eval)

        return _Step(LossEvaluatorList=loss_eval_list,
                     callbacks=None,
                     verbose=True,
                     stopreturn=float('inf'))

    def test_hasattr(self, params_func):
        assert params_func.cbs is None  # Has callbacks called cbs
        assert params_func.v  # Has verbose setting called v
        assert hasattr(params_func, '__call__')  # Has call attribute

        for i, parm in enumerate(
                inspect.signature(params_func.__call__).parameters.keys()):  # Call has correct signature
            assert parm == ('X', 'workers', 'full', '_force')[i]

    @pytest.mark.parametrize('config, float_ret', [(([1, 1],), True),
                                                   (([0, 0], 1, True, False), False)])
    def test_return(self, config, params_func, float_ret):
        if float_ret:
            assert params_func(*config) == 100
        else:
            answer = params_func(*config)
            assert len(answer) == 3
            assert all([isinstance(cont, EvaluatorReturn) for cont in answer])

    def test_wrapping(self, params_func):
        wrapped = _FunctionWrapper(params_func)

        assert hasattr(wrapped, '__call__')
        assert hasattr(wrapped, 'resids')

        params_func.cbs = lambda x: x
        with pytest.warns(UserWarning, match="Callbacks provided through the Optimization class are ignored"):
            _FunctionWrapper(params_func)


class TestReaxFFError:
    built_tasks: Dict[str, ReaxFFError] = {}

    @pytest.fixture(scope='class')
    def check_result(self, input_files):
        with (input_files / 'check_result.pkl').open('rb') as file:
            result = pickle.load(file)
        return result

    @pytest.mark.parametrize("name, factory", [('classic', ReaxFFError.from_classic_files),
                                               ('params_pkl', ReaxFFError.from_params_files),
                                               ('params_yml', ReaxFFError.from_params_files)])
    def test_load(self, name, factory, input_files, tmp_path):
        if 'params' in name:
            suffix = name.split('_')[1]
            for file in ('data_set.' + suffix, 'job_collection.' + suffix, 'reax_params.pkl'):
                shutil.copy(input_files / file, tmp_path / file)
                if file != 'reax_params.pkl':
                    with pytest.raises(FileNotFoundError):
                        factory(tmp_path)
        else:
            for file in ('control', 'ffield_bool', 'ffield_max', 'ffield_min', 'ffield_init', 'geo', 'trainset.in'):
                shutil.copy(input_files / file, tmp_path / file)
        task = factory(tmp_path)

        assert isinstance(task.dat_set, DataSet)
        assert isinstance(task.job_col, JobCollection)
        assert isinstance(task.rxf_eng, ReaxParams)
        assert isinstance(task.loss, Loss)
        assert isinstance(task.par_levels, ParallelLevels)
        assert isinstance(task.scaler, LinearParameterScaler)

        self.built_tasks[name] = task

    @pytest.mark.parametrize("method, suffix", [('save', 'yml'), ('checkpoint_save', 'pkl')])
    def test_save(self, method, suffix, tmp_path):
        if len(self.built_tasks) == 0:
            pytest.xfail("No tasks constructed successfully")

        task = self.built_tasks[[*self.built_tasks.keys()][0]]
        getattr(task, method)(tmp_path)

        for file in ('data_set.' + suffix, 'job_collection.' + suffix,
                     'reax_params.pkl' if suffix == 'pkl' else 'ffield'):
            assert Path(tmp_path, file).exists()

    @pytest.mark.parametrize("name", ['classic', 'params_pkl', 'params_yml'])
    def test_calculate(self, name, check_result):
        if name not in self.built_tasks:
            pytest.xfail("Task not constructed successfully")

        # TODO Remove when AMS bug is fixed
        if name == 'params_yml':
            pytest.xfail("Known AMS bug causes this to fail.")

        task = self.built_tasks[name]
        fx, resids, cont = check_result
        result = task._calculate([0.5] * task.n_parms)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == fx
        assert result[1] == resids
        assert all([r == c for r, c in zip(result[2], cont)])

    def test_race(self, monkeypatch, input_files):
        """ Tests calculate method with multiple calls from multiple threads to ensure there are no race conditions.
            The actual job collection evaluation is monkeypatched to return back the parameter set.
        """

        self.built_tasks['classic'] = ReaxFFError.from_classic_files(input_files)
        lock = threading.Lock()

        def mock_run(engine, *args, **kwargs):
            with lock:
                ff_file = engine.input.ReaxFF.ForceField
                loaded_eng = ReaxParams(ff_file)
                return loaded_eng.x

        def mock_evaluate(ff_results, *args, **kwargs):
            return ff_results

        monkeypatch.setattr(self.built_tasks['classic'].job_col, 'run', mock_run)
        monkeypatch.setattr(self.built_tasks['classic'].dat_set, 'evaluate', mock_evaluate)

        params_orig = np.random.uniform(size=(100, self.built_tasks['classic'].n_parms))
        with ThreadPoolExecutor(max_workers=np.clip(os.cpu_count(), 2, None)) as executor:
            params_rtrn = np.array([*executor.map(self.built_tasks['classic']._calculate, params_orig)])

        params_rtrn = np.array([vector[self.built_tasks['classic'].rxf_eng.is_active] for vector in params_rtrn])

        params_orig = np.array([self.built_tasks['classic'].scaler.scaled2real(vector) for vector in params_orig])
        params_orig = np.round(params_orig, 4)
        assert np.all(params_orig == params_rtrn)
