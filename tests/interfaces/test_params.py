import inspect
import os
import pickle
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Union

import numpy as np
import pytest

pytest.importorskip('scm', reason="SCM ParAMS needed to test and use the ParAMS interface.")

from scm.params.core.opt_components import _LossEvaluator, EvaluatorReturn
from scm.params.common.parallellevels import ParallelLevels
from scm.params.core.dataset import DataSet, Loss
from scm.params.core.jobcollection import JobCollection
from scm.params.core.opt_components import _Step
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
from scm.params.parameterinterfaces.reaxff import ReaxParams

from glompo.interfaces.params import _FunctionWrapper, ReaxFFError, GlompoParamsWrapper, setup_reax_from_classic
from glompo.opt_selectors.baseselector import BaseSelector
from glompo.optimizers.baseoptimizer import BaseOptimizer
from glompo.common.namedtuples import Result
from glompo.core.optimizerlogger import BaseLogger

from scm.params.common._version import __version__ as PARAMS_VERSION
PARAMS_VERSION_INFO = tuple(map(int, PARAMS_VERSION.split('.')))


class FakeLossEvaluator(_LossEvaluator):
    def check(self):
        pass

    def __call__(self, x):
        if PARAMS_VERSION_INFO == (0, 5, 0):
            er = EvaluatorReturn(100, x, self.name, self.ncalled, self.interface,
                                 None, [5, 2, -1, -9], [0.10, 0.25, 0.30, 0.35], 0)
        else:
            er = EvaluatorReturn(100, x, self.name, self.ncalled, self.interface,
                                 None, [5, 2, -1, -9], [0.10, 0.25, 0.30, 0.35], 0, 'sse')

        return er


class FakeStep(_Step):
    def __init__(self):
        self.cbs = None


class FakeReaxParams(ReaxParams):
    def __init__(self): ...

    @property
    def active(self):
        return self

    @property
    def range(self):
        return [(0, 1)]


class FakeSelector(BaseSelector):
    def select_optimizer(self, manager: 'GloMPOManager', log: BaseLogger, slots_available: int) -> \
            Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]], None, bool]: ...


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

        params_func.cbs = lambda x: x
        with pytest.warns(UserWarning, match="Callbacks provided through the Optimization class are ignored"):
            _FunctionWrapper(params_func)


def test_wrapper_run(monkeypatch):
    def mock_start_manager():
        return Result([0] * 5, 0, {}, {})

    wrapper = GlompoParamsWrapper(FakeSelector(BaseOptimizer))

    monkeypatch.setattr(wrapper.manager, 'start_manager', mock_start_manager)

    with pytest.warns(RuntimeWarning, match="The x0 parameter is ignored by GloMPO."):
        wrapper.manager.converged = True
        res = wrapper.minimize(FakeStep(), [1] * 5, [[0, 1]] * 5)

    assert isinstance(res, MinimizeResult)
    assert res.x == [0] * 5
    assert res.fx == 0
    assert res.success


class TestReaxFFError:
    @pytest.fixture(scope='class')
    def params_collection(self, input_files):
        ds, jc, re = setup_reax_from_classic(input_files)
        return {'dat_set': ds,
                'job_col': jc,
                'par_eng': re}

    @pytest.fixture(scope='function')
    def task(self, params_collection):
        ds = params_collection['dat_set'].copy()
        jc = params_collection['job_col']
        re = params_collection['par_eng'].copy()
        return ReaxFFError(ds, jc, re)

    @pytest.fixture(scope='class')
    def check_result(self, input_files):
        with (input_files / 'check_result.pkl').open('rb') as file:
            result = pickle.load(file)

        try:
            return result[PARAMS_VERSION]
        except KeyError:
            return None

    @pytest.fixture(scope='function')
    def simple_func(self, request):
        return ReaxFFError(None, None, FakeReaxParams(), request.param, False)

    @staticmethod
    def mock_calculate(x):
        default = float('inf'), np.array([float('inf')]), np.array([float('inf')])
        return default, default

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
            for file in ('control', 'ffield_bool', 'ffield_max', 'ffield_min', 'ffield', 'geo', 'trainset.in'):
                shutil.copy(input_files / file, tmp_path / file)
        task = factory(tmp_path)

        assert isinstance(task.dat_set, DataSet)
        assert isinstance(task.job_col, JobCollection)
        assert isinstance(task.par_eng, ReaxParams)
        assert isinstance(task.loss, Loss)
        assert isinstance(task.par_levels, ParallelLevels)

    def test_props(self, task):
        assert task.n_parms == 87
        assert task.n_all_parms == 701
        assert len(task.active_names) == 87
        assert len(task.active_abs_indices) == 87

    @pytest.mark.parametrize("method, suffix", [('save', 'yml'), ('checkpoint_save', 'pkl')])
    def test_save(self, method, suffix, tmp_path, task):
        getattr(task, method)(tmp_path)

        for file in ('data_set.' + suffix, 'job_collection.' + suffix,
                     'reax_params.pkl' if suffix == 'pkl' else 'ffield'):
            assert Path(tmp_path, file).exists()

    @pytest.mark.parametrize("name", ['classic', 'params_pkl', 'params_yml'])
    def test_calculate(self, name, task, check_result):
        if check_result is None:
            pytest.xfail("Calculate result check not yet supported for this ParAMS version.")

        fx, resids, cont = check_result
        result = task._calculate([0.5] * task.n_parms)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert np.isclose(result[0][0], fx, atol=1e-6)
        assert np.all(np.isclose(result[0][1], resids, atol=1e-6))
        assert np.all(np.isclose(result[0][2], cont, atol=1e-6))

    def test_race(self, monkeypatch, task, input_files):
        """ Tests calculate method with multiple calls from multiple threads to ensure there are no race conditions.
            The actual job collection evaluation is monkeypatched to return back the parameter set.
        """
        lock = threading.Lock()

        def mock_run(engine, *args, **kwargs):
            with lock:
                ff_file = engine.input.ReaxFF.ForceField
                loaded_eng = ReaxParams(ff_file)
                return loaded_eng.x

        def mock_evaluate(ff_results, *args, **kwargs):
            return None, ff_results, None

        monkeypatch.setattr(task.job_col, 'run', mock_run)
        monkeypatch.setattr(task.dat_set, 'evaluate', mock_evaluate)

        params_orig = np.random.uniform(size=(100, task.n_parms))
        with ThreadPoolExecutor(max_workers=np.clip(os.cpu_count(), 2, None)) as executor:
            params_rtrn = np.array(
                [*executor.map(lambda x: task._calculate(x)[0][1], params_orig)])

        params_rtrn = np.array([vector[task.par_eng.is_active] for vector in params_rtrn])

        params_orig = np.array([task.convert_parms_scaled2real(vector) for vector in params_orig])
        params_orig = np.round(params_orig, 4)
        assert np.all(params_orig == params_rtrn)

    @pytest.mark.parametrize('simple_func', [None, DataSet()], indirect=['simple_func'])
    def test_detailed_call(self, simple_func, monkeypatch):
        monkeypatch.setattr(simple_func, '_calculate', self.mock_calculate)

        res = simple_func.detailed_call([0.5])
        expected = (float('inf'), np.array([float('inf')]))

        if simple_func.val_set is not None:
            expected *= 2

        assert res == expected

    def test_indices_transform(self, task):
        abs_ind = [5, 70, 43, 26, 87, 124, 677, 656]
        rel_ind = [0, 3, 2, 1, 4, 5, 7, 6]
        activate = [False] * 701

        for t in abs_ind:
            activate[t] = True

        task.par_eng.is_active = activate

        assert task.convert_indices_rel2abs(rel_ind) == abs_ind
        assert task.convert_indices_abs2rel(abs_ind) == rel_ind

    @pytest.mark.parametrize('val', [0, 0.1, 0.5, 1])
    def test_parms_transforms(self, val, task):
        for n in (87, 701):
            vec = [val] * n
            assert np.isclose(task.convert_parms_real2scaled(task.convert_parms_scaled2real(vec)), vec).all()

    def test_parms_transforms_raises(self, task):
        with pytest.raises(ValueError, match="Cannot parse x with length"):
            task.convert_parms_real2scaled([0.5] * 100)

    @pytest.mark.parametrize('nparams, full', [(701, True),
                                               (87, False)])
    def test_set_parameters(self, nparams, full, task):
        new = [0.5] * nparams

        with pytest.warns(UserWarning, match="x contains parameters which are outside their bounds."):
            task.set_parameters(new, 'scaled', full)
        task_parms = task.par_eng.x if full else task.par_eng.active.x

        assert np.isclose(task_parms, task.convert_parms_scaled2real(new)).all()

    @pytest.mark.parametrize('simple_func', [None, DataSet()], indirect=['simple_func'])
    def test_resids(self, simple_func, monkeypatch):
        monkeypatch.setattr(simple_func, '_calculate', self.mock_calculate)
        res = simple_func.resids([0.5])
        assert res == np.array([float('inf')])

    @pytest.mark.parametrize('activate', [range(5),
                                          ('O.H.S:-p_hb2;;18;;Hydrogen bond/bond order',
                                           'C.S.S.C:V_3;;16a;;V3-torsion barrier',)])
    def test_toggle_parameter(self, task, activate):
        task.toggle_parameter(range(701), toggle='off')
        assert task.n_parms == 0

        task.toggle_parameter(activate, toggle='on')
        assert task.n_parms == len(activate)

    @pytest.mark.parametrize('force', [True, False])
    def test_toggle_parameter_notallowed(self, task, force):
        task.toggle_parameter(range(701), False)
        with pytest.warns(UserWarning, match="The following parameters should never be activated:"):
            task.toggle_parameter(('O.H.S:-p_hb2;;18;;Hydrogen bond/bond order',
                                   '0.O.S.0:n/a 1;;n/a;;n/a',
                                   '0.H.S.0:n/a 1;;n/a;;n/a',
                                   'C.S.S.C:V_3;;16a;;V3-torsion barrier',),
                                  toggle='on',
                                  force=force)
        assert task.n_parms == 4 if force else 2

    def test_toggle_parameter_warning(self, task):
        with pytest.warns(UserWarning, match="not recognised, ignoring."):
            task.toggle_parameter(['randomstring name'], 'off')

    @pytest.mark.parametrize('activate, weight', [(range(100), 1),
                                                  (('forces("dmds-CS10", 1, 0)',
                                                    'forces("dmds-SS1.7", 8, 2)',
                                                    'distance("dmds",5,0)',
                                                    'angles("dmds",(2,3,8))',), 0.2),
                                                  ({4: 1, 132: 5, 242: 2.2, 2342: 0.3332, 543: 8}, None)])
    def test_reweigh_residuals(self, activate, task, weight):
        task.scale_residuals = True

        task.reweigh_residuals(range(4875), 0)
        assert sum([d.weight for d in task.dat_set]) == 0

        task.reweigh_residuals(activate, weight)

        if weight is None:
            tot_weight = sum(activate.values())
        else:
            tot_weight = len(activate) * weight

        resids = task.detailed_call([0.5] * 87)[1]
        assert sum([d.weight for d in task.dat_set]) == tot_weight
        assert sum(map(bool, resids)) == len(activate)

    def test_reweigh_residuals_warning(self, task):
        with pytest.raises(ValueError, match="new_weight cannot be None if resids is a sequence of names or indices."):
            task.reweigh_residuals([0, 1, 2])


@pytest.mark.skipif(PARAMS_VERSION_INFO <= (0, 5, 0))
class TestOptimizationWrapper:
    pass
