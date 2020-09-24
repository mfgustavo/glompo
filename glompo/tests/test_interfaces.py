import inspect

import pytest

try:
    from scm.params.core.opt_components import _Step, _LossEvaluator, EvaluatorReturn

    HAS_PARAMS = True
except (ModuleNotFoundError, ImportError):
    HAS_PARAMS = False

from glompo.interfaces.params import _FunctionWrapper


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
