import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tables as tb
from scm.params.common._version import __version__ as PARAMS_VERSION
from scm.params.common.helpers import plams_initsettings
from scm.params.common.parallellevels import ParallelLevels
from scm.params.common.reaxff_converter import geo_to_params, trainset_to_params
from scm.params.core.callbacks import Callback
from scm.params.core.dataset import DataSet, SSE
from scm.params.core.jobcollection import JobCollection
from scm.params.core.lossfunctions import LOSSDICT, Loss
from scm.params.core.opt_components import LinearParameterScaler, _LossEvaluator, _Step
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
from scm.params.parameterinterfaces.base import BaseParameters, Constraint
from scm.params.parameterinterfaces.reaxff import ReaxParams  # Instead of ReaxFFParameters for backward compatibility
from scm.params.parameterinterfaces.xtb import XTBParams
from scm.plams.core.errors import ResultsError
from scm.plams.core.functions import config, finish, init
from scm.plams.core.jobrunner import JobRunner
from scm.plams.interfaces.adfsuite.reaxff import reaxff_control_to_settings

from ..core.manager import GloMPOManager
from ..opt_selectors.baseselector import BaseSelector

try:
    from scm.params.core.dataset import DataSetEvaluationError
except ImportError:
    # Different versions of ParAMSs raise different error types.
    DataSetEvaluationError = ResultsError

__all__ = ("GlompoParamsWrapper",
           "ReaxFFError",
           "XTBError",
           "setup_reax_from_classic",
           "setup_reax_from_params",
           "setup_xtb_from_params",)

PARAMS_VERSION_INFO = tuple(map(int, PARAMS_VERSION.split('.')))


class _FunctionWrapper:
    """ Wraps function produced by ParAMS internals (instance of :class:`scm.params.core.opt_components._Step`) to
    match the API required by the :attr:`.GloMPOManager.task`. Can be modified to achieve compatibility with other
    optimizers.
    """

    def __init__(self, func: _Step):
        self.func = func
        if self.func.cbs:
            warnings.warn("Callbacks provided through the Optimization class are ignored. Callbacks to individual "
                          "optimizers can be passed to GloMPO through BaseSelector objects. Callbacks to control the "
                          "manager itself are passed using GloMPO BaseChecker objects, some conditions should be sent "
                          "as BaseHunter objects.", UserWarning)
            self.func.cbs = None

    def __call__(self, pars) -> float:
        return self.func(pars)


class GlompoParamsWrapper(BaseOptimizer):
    """ Wraps the GloMPO manager into a ParAMS :class:`~scm.params.optimizers.base.BaseOptimizer`.
    This is not the recommended way to make use of the GloMPO interface, it is preferable to make use of the
    :class:`.BaseParamsError` classes. This class is only applicable in cases where the ParAMS
    :class:`~scm.params.core.parameteroptimization.Optimization` class interface is preferred.

    Parameters
    ----------
    opt_selector
        Initialised :class:`.BaseSelector` object which specifies how optimizers are selected and initialised.
    **manager_kwargs
        Optional arguments to the :class:`.GloMPOManager` initialisation function.

    Notes
    -----
    `manager_kwargs` accepts all arguments of :meth:`.GloMPOManager.setup` but required GloMPO arguments
    :attr:`~.GloMPOManager.task` and :attr:`~.GloMPOManager.bounds` will be overwritten as they are passed by the
    :meth:`minimize` function in accordance with ParAMS API.
    """

    def __init__(self, opt_selector: BaseSelector, **manager_kwargs):
        self.manager = GloMPOManager()
        self.manager_kwargs = manager_kwargs
        for kw in ['task', 'bounds']:
            if kw in self.manager_kwargs:
                del self.manager_kwargs[kw]

        self.selector = opt_selector

    def minimize(self,
                 function: _Step,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 workers: int = 1) -> MinimizeResult:
        """
        Passes 'function' to GloMPO to be minimized. Returns an instance of MinimizeResult.

        Parameters
        ----------
        function
            Function to be minimized, this is passed as GloMPO's :attr:`~.GloMPOManager.task` parameter.
        x0
            Ignored by GloMPO, the correct way to control the optimizer starting points is by using GloMPO
            :class:`.BaseGenerator` objects.
        bounds
            Sequence of (min, max) pairs used to bound the search area for every parameter. The 'bounds' parameter is
            passed to GloMPO as its :attr:`~.GloMPOManager.bounds` parameter.
        workers
            Represents the maximum number of optimizers run in parallel. Passed to GloMPO as its
            :attr:`~.GloMPOManager.max_jobs` parameter if it has not been sent during initialisation via
            `manager_kwargs` otherwise ignored. If allowed to default this will usually result in the number of
            optimizers as there are cores available.

        Notes
        -----
        GloMPO is not currently compatible with using multiple :class:`~scm.params.core.dataset.DataSet` and only the
        first one will be considered.

        By default ParAMS shifts and scales all parameters to the interval (0, 1). GloMPO will work in this space and be
        blind to the true bounds, thus results from the GloMPO logs cannot be applied directly to the function.
        """

        warnings.warn("The x0 parameter is ignored by GloMPO. To control the starting locations of optimizers within "
                      "GloMPO make use of its BaseGenerator objects.", RuntimeWarning)

        if 'max_jobs' not in self.manager_kwargs:
            self.manager_kwargs['max_jobs'] = workers

        # Silence function printing
        function.v = False

        self.manager.setup(task=_FunctionWrapper(function), bounds=bounds, opt_selector=self.selector,
                           **self.manager_kwargs)

        result = self.manager.start_manager()

        # Reshape glompo.common.namedtuples.Result into scm.params.optimizers.base.MinimizeResult
        params_res = MinimizeResult()
        params_res.x = result.x
        params_res.fx = result.fx
        params_res.success = self.manager.converged and len(result.x) > 0

        return params_res

    def reset(self):
        self.manager = GloMPOManager()


class Optimization:
    """
    Parameters
    ----------
    jobcollection
        Job collection holding all jobs necessary to evaluate the `datasets`
    datasets
        Data Set(s) to be evaluated. In the most simple case, one data set will be evaluated as the training set.
        Multiple data sets can be passed to be evaluated sequentially at every optimizer step. In this case, the first
        data set will be interpreted as the training set, the second as a validation set.
    parameterinterface
        The interface to the parameters that are to be optimized.
    optimizer
        IGNORED.
    title : optional, str
        The working directory for this optimization. Once :meth:`optimize` is called, will switch to it.
        (see `glompo_kwargs`)
    plams_workdir_path : optional, str
        The folder in which the PLAMS working directory is created. By default the PLAMS working directory is created
        inside of `$SCM_TMPDIR` or `/tmp` if the former is not defined. When running on a compute cluster this variable
        can be set to a local directory of the machine where the jobs are running, avoiding a potentially slow PLAMS
        working directory that is mounted over the network.
    validation
        If the passed value is :code:`0<float<1`, a validation set will be created from a `validation` percentage of the
        first data set in `datasets`. If the passed value is :code:`1<float<len(datasets[0])`, will create a validation
        set with `validation` entries taken from the first data set in `datasets`. If you would like to pass a
        :class:`~scm.params.core.dataset.DataSet` instance instead, you can do so in the `datasets` parameter.
    callbacks
        IGNORED.
    constraints
        Additional constraints for candidate solutions of :math:`\\mathbf{x}^*`. If the any of these return
        :obj:`False`, the solution will not be considered.
    parallel
        Configuration for the parallelization at all levels of a parameter optimization.
    verbose
        Print the current best loss function value each time we improve
    skip_x0
        Before an optimization process starts, a DataSet will be evaluated with the initial parameters
        :math:`\\mathbf{x}_0`. If this initial evaluation returns an infinite loss function value, will raise an error
        by default. This behavior is expecting that the initial parameters are generally valid and the cause of the
        non-finite loss is probably due to bad :class:`.plams.Settings` of an entry in the `jobcollectgion``. However,
        if it is not known if the initial parameters can be trusted or raising an error is not desired for other
        reasons, this parameter can be set to :obj:`True` to skip the initial evaluation.
    logger_every
        See `every_n_iter` in :class:`~scm.params.core.callbacks.Logger`. This option is ignored if a Logger is provided in the callbacks.

    :Per Data Set Parameters:

    .. note::

        The following parameters will be applied to all entries in `datasets`, meaning each Data Set will
        be evaluated with the same settings.
        To override this, any of the parameters below can also take a list with the same number of elements as
        :code:`len(datasets)`, mapping individual settings to every `datasets` entry.

    loss
        A :ref:`Loss Function <Loss Functions>` instance to compute the loss of every new parameter set. Residual Sum of
        Squares by default.
    batch_size
        The number of entries to be evaluated per epoch. If :obj:`None`, all entries will be evaluated.

        .. note::

           One job calculation can have multiple property entries in a training set (`e.g. Energy and Forces), thus,
           this parameter is not the same as as `maxjobs`.

           If both, `maxjobs` and `batch_size` are set, the former will be applied first.  If the resulting set is
           still larger than `batch_size`, will apply filtering by `batch_size`.

    use_pipe
        Whether to use the :class:`~.AMSWorker` interface for suitable jobs.
    dataset_names
        When evaluating multiple `datasets`, can be set to give each entry a name. Possible logger callbacks will create
        and write data into this subdirectory.
        Defaults to ``['trainingset', 'validationset', 'dataset03', ..., 'datasetXX']``
    eval_every
        Evaluate the Data Set at every `eval_every` call.

        .. warning::

            The first entry in `datasets` represents the training set and must be evaluated at every call.
            It's frequency will always be `1`.

    maxjobs
        Whether to limit each Data Set evaluation to a subset of maximum `maxjobs`. Igonored if :obj:`None`.
    maxjobs_shuffle
        If `maxjobs` is set, will generate a new subset of the Data Set with `maxjobs` at every evaluation.
    **glompo_kwargs
        GloMPO related arguments sent to :meth:`GloMPOManager.setup()`.

        The following extra keywords are allowed:

        :code:`'scaler'`
           Extra keyword which specifies the type of scaling used by function. Defaults to a linear scaling of all
           parameters between 0 and 1.

        The following keywords will be ignored if provided:

        :code:`'bounds'`
           Automatically extracted from `parameterinterface`.

        :code:`'task'`
           It is constructed within this class from `jobcollection`, `dataset`, `parameterinterface`.

        :code:`'working_dir'`
           `title` will be used as this parameter.

        :code:`'overwrite_existing'`
           No overwriting allowed according to ParAMS behavior. `title` will be incremented until a non-existent
           directory is found.

        :code:`'max_jobs'`
           Will be calculated from `parallel`.

        :code:`'backend'`
           Only :code:`'threads'` are allowed within ParAMS.


    """

    def __init__(self,
                 jobcollection: JobCollection,
                 datasets: Union[DataSet, Sequence[DataSet]],
                 parameterinterface: BaseParameters,
                 optimizer: BaseOptimizer,
                 title: str = 'opt',
                 plams_workdir_path: Optional[str] = None,
                 validation: Optional[float] = None,
                 callbacks: Sequence[Callback] = None,
                 constraints: Sequence[Constraint] = None,
                 parallel: ParallelLevels = None,
                 verbose: bool = True,
                 skip_x0: bool = False,
                 logger_every: Union[dict, int] = None,
                 # Per dataset settings
                 loss: Union[Loss, Sequence[Loss]] = 'sse',
                 batch_size: Union[int, Sequence[int]] = None,
                 use_pipe: Union[bool, Sequence[bool]] = True,
                 dataset_names: Sequence[str] = None,
                 eval_every: Union[int, Sequence[int]] = 1,
                 maxjobs: Union[None, Sequence[int]] = None,
                 maxjobs_shuffle: Union[bool, Sequence[bool]] = False,
                 **glompo_kwargs):
        assert isinstance(jobcollection,
                          JobCollection), f"JobCollection instance not understood: {type(jobcollection)}."
        self.jobcollection = jobcollection

        assert issubclass(parameterinterface.__class__, BaseParameters), \
            f"Parameter interface type not understood: " \
            f"{type(parameterinterface)}. Must be a subclass of BaseParameters."
        self.interface = parameterinterface

        assert len(self.interface.active) > 0, \
            "The parameter interface does not contain any parameters marked for optimization. Check the " \
            "`interface.is_active` attribute and make sure that at least one parameter is marked as active."
        for p in self.interface.active:
            assert p.range[0] <= p.value <= p.range[1], \
                f"The ranges or value of {repr(p)} are ill-defined. " \
                f"Please make sure that `range[0] <= value <= range[1]`."

        # Warnings about GloMPO / ParAMS interface differences
        if callbacks:
            print("WARNING: Callbacks are ignored.")
        print("WARNING: The 'optimizer' argument is ignored. Please send optimizer related arguments to glompo_kwargs")

        # Scaler must be GloMPO-wide.
        # Enforce [0,1] scaling by default, other scalers can be requested in glompo_kwargs['scaler']
        self.scaler = LinearParameterScaler(self.interface.active.range)

        # Check loss
        def checkloss(loss, pre=''):
            if isinstance(loss, str):
                assert loss.lower() in LOSSDICT, f"The requested loss '{loss}' is not known."
                loss = LOSSDICT[loss.lower()]()
            assert isinstance(loss, Loss), f"{pre} Loss argument must be a subclass of the `Loss` class."
            return loss

        loss = [checkloss(i) for i in loss] if isinstance(loss, List) else checkloss(loss)

        if constraints is not None:
            assert all(isinstance(i, (Constraint, Constraint._Operator)) for i in constraints), \
                f"Unexpected constraints class provided: {', '.join([str(i.__class__) for i in constraints])}."
            if not all(c(self.interface) for c in constraints):
                print("WARNING: Initial interface violates constraints! This might result in undesired optimizer "
                      "behaviour.")
        self.constraints = constraints

        if parallel is None:
            self.parallel = ParallelLevels()
        else:
            assert isinstance(parallel, ParallelLevels), \
                f"Type of the parallel argument is{type(parameterinterface)}, but should be ParallelLevels."
            self.parallel = parallel

        self.objective = self._wrap_datasets(datasets, validation, loss, batch_size, use_pipe, dataset_names,
                                             eval_every, maxjobs, maxjobs_shuffle)

        self.plams_workdir_path = plams_workdir_path or os.getenv('SCM_TMPDIR', '/tmp')
        self.title = self.mkdirs(title)
        self.skip_x0 = skip_x0
        self.verbose = verbose

        self.glompo = GloMPOManager()

    def optimize(self) -> MinimizeResult:
        """ Start the optimization given the initial parameters. """
        working_dir = Path(self.title)
        if working_dir.exists():
            i = 1
            while working_dir.with_suffix(f"{i:03}").exists():
                i += 1
            print(f"'{working_dir}' already exists. Will use '{working_dir.with_suffix(f'{i:03}')}' instead.")
            working_dir = working_dir.with_suffix(f'{i:03}')

        idir = working_dir / 'settings_and_initial_data'
        idir.mkdir(parents=True, exist_ok=True)

        self.interface.pickle_dump(idir / 'initial_parameter_interface.pkl')
        self.jobcollection.pickle_dump(idir / 'jobcollection.pkl.gz')
        self.jobcollection.store(idir / 'jobcollection.yaml.gz')

        (idir / 'datasets').mkdir(parents=True, exist_ok=True)
        for obj in self.objective:
            obj.dataset.pickle_dump(idir / 'datasets' / obj.name + '.pkl.gz')
            obj.dataset.store(idir / 'datasets' / obj.name + '.yaml.gz')

        init(config_settings=plams_initsettings, path=self.plams_workdir_path)
        # All parallel parameter vectors share the same job runner, so the total number of jobs we want to have
        # running is the product of the parallelism at both optimizer and job collection level! (Like this we get
        # dynamic load balancing between different parameter vectors for free.)
        # todo figure out job runner within glompo context
        config.default_jobrunner = JobRunner(parallel=True,
                                             maxjobs=self.parallel.optimizations *
                                                     self.parallel.parametervectors *
                                                     self.parallel.jobs)

        # todo setup manager here

        # Single evaluation with the initial parameters
        fx0 = float('inf')
        if not self.skip_x0:
            fx0 = self.initial_eval()
        for i in self.objective:
            i.constraints = self.constraints  # do not include constraints in initial evaluation

        # Optimization
        x0 = self.scaler.real2scaled(self.interface.active.x)
        bnds = self.scaler.bounds  # Scaler determines the bounds
        f = _Step(self.objective, None, verbose=self.verbose)  # make one callable function
        self.glompo.start_manager()

        if self.result.success:
            self.result.x = self.scaler.scaled2real(self.result.x)

        finish()

        # get the smallest trainingset fx and corresponding x, in order of priority:
        # first the initial parameters, then the ones suggested by the optimizer, then the ones that the Logger
        # logged as being the best
        fx_min = fx0
        x_min = self.scaler.scaled2real(x0)
        if self.result.fx < fx_min:
            fx_min = self.result.fx
            x_min = self.result.x
        if len(self.callbacks) > 0 and isinstance(self.callbacks[0], Logger):
            fx_min = self.callbacks[0].fx_dict.get('trainingset', self.result.fx)
            x_min = self.callbacks[0].x_dict.get('trainingset', self.result.x)

        if self.callbacks:
            [cb.on_end() for cb in self.callbacks]

        if self.result.success:
            printnow(f"Optimization done after {t()}")
            self.result.fx = fx_min
            self.result.x = x_min.copy()
            self.interface.active.x = x_min.copy()
        else:
            printerr(f"WARNING: Optimization unsuccessful after {t()}")

        with open('summary.txt', 'a') as f:
            print(f"End time:   {datetime.now()}", file=f)

        os.chdir(root)
        printnow(f"Final loss: {self.result.fx:.3e}")
        return self.result

    def initial_eval(self):
        """ Evaluate x0 before the optimization. Returns fx"""
        e = self.interface.get_engine()
        par = self.parallel.copy()
        par.jobs = self.parallel.jobs * self.parallel.parametervectors
        r = [i.evaluate(e, parallel=par, delete_failed_jobs=False) for i in self.objective]
        fx = r[0][0]  # fx value of the training set
        if not np.isfinite(fx):
            print("\n\nAborting Optimization due to initial parameters producing an infinite loss function value.")
            print("This usually indicates bad settings of a JobCollection entry causing a job to crash.")
            print("You can override this behavior by setting 'skip_x0=True' at init.\n")
            print("Entries with the following jobids have crashed:\n")
            names = [i.name for i in r[0][-1].values() if not i.ok()]  # jobresults [-1] of the training set [0]
            errms = [r[0][-1][i].get_errormsg() for i in names]
            for name, errormsg in zip(names[:10], errms[:10]):
                print('...')
                print(f"ID:    {name}")
                print(f"ERROR: {errormsg}")
            if len(names) > 10:
                print(f"--- and {len(names) - 10} more ---")
            print()
            print("You may be able to find outputs of the failed jobs in:")
            print(config.default_jobmanager.workdir)
            print()
            raise ValueError("Initial evaluation of f(x0) failed.")

        print(f"Initial loss: {fx:.3e}")
        self.glompo.opt_log.put_manager_metadata('initial_parameter_results', {'x': self.interface.active.x, 'fx': fx})

        return fx

    # def __str__(self):
    #     s = f"{self.__class__.__name__}() Instance Settings:\n"
    #     l = len(s) - 1
    #     s += l * '=' + '\n'
    #     s += strpad(os.path.abspath(self.title), 35, "Workdir:")
    #     s += strpad(len(self.jobcollection), 35, "JobCollection size:")
    #     s += strpad(self.interface.__class__.__name__, 35, "Interface:")
    #     s += strpad(len(self.interface.active), 35, "Active parameters:")
    #     s += strpad(self.optimizer.__class__.__name__, 35, "Optimizer:")
    #     s += strpad(self.parallel, 35, "Parallelism:")
    #     s += strpad(self.verbose, 35, "Verbose:")
    #     if self.callbacks:
    #         s += strpad(self.callbacks[0].__class__.__name__, 35, "Callbacks:")
    #         if len(self.callbacks) > 1:
    #             for cb in self.callbacks[1:]:
    #                 s += strpad(cb.__class__.__name__, 35)
    #     if self.constraints:
    #         s += strpad(repr(self.constraints[0]), 35, "Constraints:")
    #         if len(self.constraints) > 1:
    #             for cb in self.constraints[1:]:
    #                 s += strpad(repr(cb), 35)
    #     s += strpad(self.plams_workdir_path, 35, "PLAMS workdir path:")
    #     s += '\nEvaluators:\n'
    #     s += '-----------\n'
    #     s += ''.join(str(i) for i in self.objective)
    #     s += '==='
    #     return s

    def _wrap_datasets(self, datasets, validation, loss, batch_size, use_pipe, dataset_names, eval_every, maxjobs,
                       maxjobs_shuffle) -> List[_LossEvaluator]:
        """ Wrap all datasets and losses into the _LossEvaluator() class, wrap all _LossEvaluators into
        _OptimizerWrapper, return the objective.
        """

        # Convert to lists
        if not isinstance(datasets, List):
            datasets = [datasets]
        tolist = lambda x: len(datasets) * [x] if not isinstance(x, List) else x
        loss, batch_size, use_pipe, dataset_names, eval_every, maxjobs, maxjobs_shuffle = [tolist(i) for i in
                                                                                           [loss, batch_size, use_pipe,
                                                                                            dataset_names, eval_every,
                                                                                            maxjobs, maxjobs_shuffle]]

        # Insert a validation set at index 1
        if validation:
            if validation > 1:
                assert validation < len(datasets[
                                            0]), f"Requested validation set {validation} is larger than the training set {len(datasets[0])}"
                validation /= len(datasets[0])
            datasets[0], valset = datasets[0].split(1 - validation, validation)
            datasets.insert(1, valset)
            dataset_names.insert(1, 'validationset')
            maxjobs.insert(1, None)
            maxjobs_shuffle.insert(1, False)
            batch_size.insert(1, None)
            for i in [loss, use_pipe, eval_every]:
                i.insert(1, i[0])

        # Basic checks
        for i in [loss, batch_size, use_pipe, dataset_names, eval_every, maxjobs, maxjobs_shuffle, use_pipe]:
            assert len(i) == len(datasets), f"List sizes do not match in {i}: Not {len(datasets)}"
        assert len(set([name for name in dataset_names if name is not None])) == len(
            [name for name in dataset_names if name is not None]), "Dataset names have to be unique."
        if eval_every[0] != 1:
            eval_every[0] = 1
            print('Enforcing first Data Set\'s evaluation frequency 1')

        # enforce trainingset and validationset names
        # this changes the dataset_names list
        for i, name in enumerate(dataset_names):
            if i == 0:
                dataset_names[i] = 'trainingset'
            elif i == 1:
                dataset_names[i] = 'validationset'
            elif name is None:
                dataset_names[i] = 'dataset{:02d}'.format(i + 1)

        # Now wrap
        objective = []
        for num, (
                _ds, _loss, _batch_size, _use_pipe, _name, _eval_every, _maxjobs, _maxjobs_shuffle,
                _use_pipe) in enumerate(
            zip(datasets, loss, batch_size, use_pipe, dataset_names, eval_every, maxjobs, maxjobs_shuffle,
                use_pipe), 1):
            evaluator = _LossEvaluator(name=_name,
                                       jobcol=self.jobcollection,
                                       dataset=_ds,
                                       interface=self.interface,
                                       scaler=self.scaler,
                                       loss=_loss,
                                       parallel=self.parallel,
                                       batch_size=_batch_size,
                                       maxjobs=_maxjobs,
                                       maxjobs_shuffle=_maxjobs_shuffle,
                                       use_pipe=_use_pipe,
                                       eval_every=_eval_every,
                                       constraints=None,  # will be set after initial_eval() is called
                                       )
            assert all(i is not None for i in _ds.get(
                'reference')), f"\n\nNot all entries in {_name} have a reference value. Please set or calculate it before starting the optimization.\nSee https://www.scm.com/doc/params/components/dataset/dataset.html for help.\n"
            objective.append(evaluator)

        return objective


class BaseParamsError:
    """ Base error function instance from which other classes derive depending on the engine used e.g. ReaxFF, xTB etc.
    Primarily initialized from ParAMS objects. To initialize from files see the class methods
    :meth:`~.ReaxFFError.from_classic_files` or :meth:`~.ReaxFFError.from_params_files`.

    Parameters
    ----------
    data_set
        Reference data used to compare against force field results.

    job_collection
        AMS jobs from which the data can be extracted for comparison to the :class:`~scm.params.core.dataset.DataSet`

    parameters
        :class:`~scm.params.parameterinterfaces.base.BaseParameters` object which holds the force field values, ranges,
        engine and which parameters are active or not.

    validation_dataset
        If a validation set is being used and evaluated along with the training set, it may be added here.
        Jobs for the validation set are expected to be included in `job_collection`.

    scale_residuals
        See :attr:`scale_residuals`.

    Notes
    -----
    The class provides several convenience functions to access/read/modify the force field parameters (for example:
    :attr:`n_parms`, :attr:`active_names`, :meth:`set_parameters`, :meth:`reweigh_residuals` etc.). These are typically
    light wrappers around various :attr:`par_eng` commands. Not all forms of interface have been provided and, in
    general, the user may access the :attr:`par_eng` directly for fine control.

    Attributes
    ----------
    dat_set : ~scm.params.core.dataset.DataSet
        Represents the training set.

    job_col : ~scm.params.core.jobcollection.JobCollection
        Represents the jobs from which model results will be extracted and compared to the training set.

    loss : Union[str, ~scm.params.core.dataset.Loss]
        Method by which individual errors are grouped into a single error function value.

    par_eng : ~scm.params.parameterinterfaces.base.BaseParameters
        Parameter engine interface representing the model and its parameters to tune.

    par_levels : ~scm.params.common.parallellevels.ParallelLevels
        The layers of parallelism possible within the evaluation of the jobs.

    scale_residuals : bool
        If :obj:`True` then the raw residuals (i.e. the differences between engine evaluation and training data)
        will be scaled by the weight and sigma values in the datasets i.e. :code:`r_scaled = weight * (r / sigma) ** 2`.
        Otherwise the raw residual is returned. This setting effects :meth:`resids` and :meth:`detailed_call`.

    val_set : ~scm.params.core.dataset.DataSet
        Optional validation set to evaluate in parallel to the training set.
    """

    def __init__(self, data_set: DataSet, job_collection: JobCollection, parameters: BaseParameters,
                 validation_dataset: Optional[DataSet] = None,
                 scale_residuals: bool = False):
        self.dat_set = data_set
        self.job_col = job_collection
        self.par_eng = parameters
        self.val_set = validation_dataset

        self.scale_residuals = scale_residuals

        self.loss = SSE()
        self.par_levels = ParallelLevels(jobs=1)

    @property
    def n_parms(self) -> int:
        """ Returns the number of active parameters.

        See Also
        --------
        :attr:`n_all_parms`
        """
        return len(self.par_eng.active.x)

    @property
    def n_all_parms(self) -> int:
        """ Returns the total number of active and inactive parameters.

        See Also
        --------
        :attr:`n_parms`
        """
        return len(self.par_eng.is_active)

    @property
    def active_abs_indices(self) -> List[int]:
        """ Returns the absolute index number of the active parameters.

        See Also
        --------
        :meth:`active_names`
        :meth:`convert_indices_abs2rel`
        :meth:`convert_rel2abs_indices`
        """
        return [p._id for p in self.par_eng.active]

    @property
    def active_names(self) -> List[str]:
        """ Returns the names of the active parameters.

        See Also
        --------
        :meth:`active_abs_indices`
        :meth:`convert_indices_abs2rel`
        :meth:`convert_rel2abs_indices`
        """
        return self.par_eng.active.names

    @property
    def bounds(self) -> Sequence[Tuple[float, float]]:
        """ Returns the min, max bounds in each dimension in **scaled space** i.e. a list of :code:`(0, 1)` tuples for
        each parameter.

        See Also
        --------
        :meth:`convert_parms_real2scaled`
        :meth:`convert_parms_scaled2real`
        """
        return [(0, 1)] * self.n_parms

    def __call__(self, x: Sequence[float]) -> float:
        """ Returns the error value between the the force field with the given parameters and the training values.

        Notes
        -----
        Optimizations are done in scaled space to improve the numerics of the problem. Thus `x` is expected to be given
        in **scaled space**. To transform from one space to another see :meth:`convert_parms_real2scaled` and
        :meth:`convert_parms_scaled2real`.
        """
        return self._calculate(x)[0][0]

    def detailed_call(self, x: Sequence[float]) -> Union[Tuple[float, np.ndarray],
                                                         Tuple[float, np.ndarray, float, np.ndarray]]:
        """ A full return of the error results.
        Returns a tuple of:

        .. code-block:: python

            training_set_error, [training_set_residual_1, ..., training_set_residual_N]

        If a validation set is included then returned tuple is:

        .. code-block:: python

           training_set_error, [training_set_residual_1, ..., training_set_residual_N],
           validation_set_error, [validation_set_residual_1, ..., validation_set_residual_N]

        See Also
        --------
        :meth:`__call__`
        """
        calc = self._calculate(x)
        ts_fx = calc[0][0]
        ts_resids = calc[0][1]
        ts_resids = self._scale_residuals(ts_resids, self.dat_set) if self.scale_residuals else ts_resids

        if self.val_set is not None:
            vs_fx = calc[1][0]
            vs_resids = calc[1][1]
            vs_resids = self._scale_residuals(vs_resids, self.val_set) if self.scale_residuals else vs_resids
            return ts_fx, ts_resids, vs_fx, vs_resids

        return ts_fx, ts_resids

    def headers(self) -> Dict[str, tb.Col]:
        """ Returns a the column headers for the :meth:`detailed_call` return.
        See :meth:`.BaseFunction.headers`.
        """
        heads = {'fx': tb.Float64Col(pos=0),
                 'resids_ts': tb.Float64Col((1, len(self.dat_set)), pos=1)}

        if self.val_set:
            heads['fx_vs'] = tb.Float64Col(pos=2)
            heads['resids_vs'] = tb.Float64Col((1, len(self.val_set)), pos=3)

        return heads

    def resids(self, x: Sequence[float]) -> np.ndarray:
        """ Method for compatibility with GFLS optimizer.
        Returns the signed differences between the force field and training set residuals. Will be scaled by sigma and
        weight if :attr:`scale_residuals` is :obj:`True`, otherwise not.
        """
        residuals = self._calculate(x)[0][1]
        if self.scale_residuals:
            residuals = self._scale_residuals(residuals, self.dat_set)

        return residuals

    def save(self, path: Union[Path, str], filenames: Optional[Dict[str, str]] = None,
             parameters: Optional[Sequence[float]] = None):
        """ Writes the :attr:`dat_set` and :attr:`job_col` to YAML files.
        Writes the engine object to an appropriate parameter file.

        Parameters
        ----------
        path
            Path to directory in which files will be saved.
        filenames
            Custom filenames for the written files. The dictionary may include any/all of the keys in the example
            below. This example contains the default names used if not given::

                {'ds': 'data_set.yml', 'jc': 'job_collection.yml', 'ff': 'ffield'}

        parameters
            Optional parameters to be written into the force field file. If not given, the parameters currently
            therein will be used.
        """
        path = Path(path).resolve(True)

        if not filenames:
            filenames = {}

        names = {'ds': filenames['ds'] if 'ds' in filenames else 'data_set.yml',
                 'jc': filenames['jc'] if 'jc' in filenames else 'job_collection.yml',
                 'ff': filenames['ff'] if 'ff' in filenames else 'ffield'}

        self.dat_set.store(str(path / names['ds']))
        self.job_col.store(str(path / names['jc']))
        self.par_eng.write(str(path / names['ff']), parameters)

    def set_parameters(self, x: Sequence[float], space: str, full: bool = False):
        """ Store parameters in the class.

        Parameters
        ----------
        x
            Parameters to store in :class:`~scm.params.parameterinterfaces.base.BaseParameters`.
        space
            Represents the space in which `x` is given. Accepts:

            #. :code:`'real'`: Actual parameter values

            #. :code:`'scaled'`: Transformed parameter values, bounded by 0 and 1 according to their ranges
               (see :meth:`convert_parms_real2scaled` and :meth:`convert_parms_scaled2real`).
        full
            If :obj:`True`, `x` is expected to be an array of ALL parameters in the force field, otherwise `x` is
            expected to be an array of active parameters only.

        Warns
        -----
        UserWarning
            If any value in `x` is outside of the bounds associated with that parameter.

        See Also
        --------
        :meth:`convert_parms_real2scaled`
        :meth:`convert_parms_scaled2real`
        """
        if space == 'scaled':
            x = self.convert_parms_scaled2real(x)
        elif space != 'real':
            raise ValueError(f"Cannot parse space='{space}', 'real' or 'scaled' expected.")

        if any([not (min_ < x_ < max_) for x_, (min_, max_) in
                zip(x, self.par_eng.range if not full else self.par_eng.active.range)]):
            warnings.warn("x contains parameters which are outside their bounds.", UserWarning)

        if full:
            self.par_eng.x = x
        else:
            self.par_eng.active.x = x

    def convert_indices_abs2rel(self, indices: List[int]) -> List[int]:
        """ Converts a sequence of absolute indices to relative indices pointing to the corresponding parameter in the
        active subset.

        Parameters
        ----------
        indices
            Sequence of absolute indices for *active* parameters.

        Returns
        -------
        List[int]
            List of the same length as `indices` with corresponding elements giving the index of the parameters in the
            smaller active subset.

        Warns
        -----
        UserWarning
            If indices contains an index for an inactive parameter. :obj:`None` will be returned for that index.

        Examples
        --------
        Suppose :attr:`par_eng` has 100 parameters of which 5 are active. The absolute index numbers of these five are:

        >>> active = [23, 57, 78, 10, 98]

        Converting to the relative indices in the active subset:

        >>> err.convert_indices_abs2rel(active)
        [1, 2, 3, 0, 4]

        Note that this method correctly accounts for the ordering of the parameters given to `indices`.

        Suppose you attempted to convert a parameter which was not active:

        err.convert_indices_abs2rel([23, 57, 1])
        [1, 2, None]
        """
        asked_names = [self.par_eng[i].name for i in indices]
        name_rel_map = {n: i for i, n in enumerate(self.par_eng.active.names)}
        return [name_rel_map[n] if n in name_rel_map else None for n in asked_names]

    def convert_indices_rel2abs(self, indices: List[int]) -> List[int]:
        """ Converts a sequence of relative indices in the active parameter subset to absolute indices in the
        :attr:`par_eng`.

        Parameters
        ----------
        indices
            Sequence of relative indices in the active parameter subset.

        Returns
        -------
        List[int]
            List of the same length as `indices` with corresponding elements giving the index of the parameters in the
            :attr:`par_eng`.

        Examples
        --------
        Suppose :attr:`par_eng` has 100 parameters of which 5 are active. To find the absolute index numbers of all of
        them:

        >>> err.convert_rel2abs_indices(range(4))
        [10, 23, 57, 78, 98]
        """
        rel_name_map = {i: n for i, n in enumerate(self.par_eng.active.names)}
        asked_names = [rel_name_map[i] for i in indices]
        return [self.par_eng._name_to_allidx[n] for n in asked_names]

    def convert_parms_real2scaled(self, x: List[float]) -> np.ndarray:
        """ Transforms parameters from their actual values, to values between 0 and 1 where 0 and 1 represent the lower
        and upper bounds of the parameter respectively.

        .. important::

           Active parameter values exist in in two spaces:

           #. The real and actual parameter values which appear in the force field.

           #. A scaled space between 0 and 1 in all dimensions where 0 and 1 represent the lower bound and upper bounds
              of the active parameters respectively.

           Optimizations are done in scaled space to improve the numerics of the problem.

        Parameters
        ----------
        x
            Sequence of parameter values to transform. May be the same length as the number of active parameters, or the
            length of the total number of parameters in the set.

        Raises
        ------
        ValueError
            If the length of `x` does not match the number of active or total parameters
        """
        min_, max_ = self._convert_parms_core(x)
        return (np.array(x) - min_) / (max_ - min_)

    def convert_parms_scaled2real(self, x: List[float]) -> np.ndarray:
        """ Transforms parameters from their [0, 1] scaled values, to actual parameter values.
        Exact opposite transformation of :meth:`convert_parms_real2scaled`.
        """
        min_, max_ = self._convert_parms_core(x)
        return (max_ - min_) * np.array(x) + min_

    def toggle_parameter(self, parameters: Union[Sequence[int], Sequence[str]], toggle: Union[str, bool] = None):
        """ De/Activate parameters.
        This means either allowing them to be changed during an optimization, or fixing their value so that they are not
        changed.

        Parameters
        ----------
        parameters
            Sequence of integers (which refer to the parameters' indices in
            :class:`~scm.params.parameterinterfaces.base.BaseParameters`) or parameter name strings which should be
            de/activated. A mix of integers and strings is not supported.
        toggle
            Accepts :code:`'on'`, :code:`'off'`, :obj:`True` or :obj:`False`. Specifies how the toggle should be
            applied. Must be supplied. :code:`'on'` means the parameters will be optimized and changed during the
            optimization. :code:`'off'` means the parameters will be fixed. To set the parameter values see
            :meth:`set_parameters`.

        Notes
        -----
        If using integers in `parameters` these are the absolute index numbers of the full parameter set. **Not** the
        parameter indices of the already activated subset. This may lead to unexpected results. For example, if you
        have a field with five activated parameters, attempting :code:`err.toggle_parameters(4, 'off')` will not
        deactivate the fifth active parameter but rather the parameter indexed 4 in the overall set. See
        :meth:`convert_indices_abs2rel` and :meth:`convert_indices_rel2abs` to be able to convert between the reference
        systems.

        Warnings
        --------
        When toggling parameters on, make sure that their associated bounds are sensible!

        See Also
        --------
        :attr:`active_abs_indices`
        :attr:`convert_indices_abs2rel`
        :attr:`convert_rel2abs_indices`
        :meth:`set_parameters`
        """

        if (toggle == 'on') or (toggle is True):
            toggle = True
        elif (toggle == 'off') or (toggle is False):
            toggle = False
        else:
            raise ValueError("Must specify toggle parameter. Do you want these parameters turned 'on' or 'off'?")

        if isinstance(parameters[0], str):
            mapping = dict(zip(self.par_eng.names, self.par_eng.is_active))
        else:
            mapping = dict(zip(range(len(self.par_eng.is_active)), self.par_eng.is_active))

        for p in parameters:
            if isinstance(p, str) and p not in mapping:
                warnings.warn(f"Parameter name '{p}' not recognised, ignoring.", UserWarning)
                continue

            mapping[p] = toggle

        self.par_eng.is_active = [*mapping.values()]

    def reweigh_residuals(self, resids: Union[Sequence[str], Sequence[int], Dict[Union[str, int], float]],
                          new_weight: Optional[float] = None):
        """ Changes weights for elements in the :class:`~scm.params.core.dataset.DataSet`.
        Can be used to `deactivate` contributions to the training set by setting their weight to zero.

        .. note::

           Deactivating a residual does not stop its associated jobs from still being calculated.

        Parameters
        ----------
        resids
            Sequence of integers (which refer to the :class:`~scm.params.core.dataset.DataSetEntry` indices in the
            :class:`~scm.params.core.dataset.DataSet`) or strings corresponding to
            :class:`~scm.params.core.dataset.DataSet` keys. A mix of integers and strings is not supported. May also be
            a dictionary mapping the above to new weight values.
        new_weight
            New weight to apply to all elements in `resids`. Ignored if `resids` is a dictionary, must be supplied
            otherwise.
        """
        ret = lambda r: new_weight  # Default new weight
        if isinstance(resids, dict):
            ret = lambda r: resids[r]
        elif new_weight is None:
            raise ValueError("new_weight cannot be None if resids is a sequence of names or indices.")

        for r in resids:
            self.dat_set[r].weight = ret(r)

    def _calculate(self, x: Sequence[float]) -> Sequence[Tuple[float, np.ndarray, np.ndarray]]:
        """ Core calculation function, returns both the error function value and the residuals. """
        default = (float('inf'), np.array([float('inf')]), np.array([float('inf')]))
        try:
            engine = self.par_eng.get_engine(self.convert_parms_scaled2real(x))
            ff_results = self.job_col.run(engine.settings, parallel=self.par_levels)
            ts_result = self.dat_set.evaluate(ff_results, self.loss, True)
            vs_result = self.val_set.evaluate(ff_results, self.loss, True) if self.val_set is not None else default
            return (ts_result[0], np.squeeze(ts_result[1]), np.squeeze(ts_result[2])), \
                   (vs_result[0], np.squeeze(vs_result[1]), np.squeeze(vs_result[2]))
        except (ResultsError, DataSetEvaluationError):
            return default, default

    @staticmethod
    def _scale_residuals(resids: np.ndarray, data_set: DataSet) -> np.ndarray:
        """ Scales a sequence of residuals by weight and sigma values in the associated
        :class:`scm.params.core.dataset.DataSet`.

        .. math::

            r_i = w_i \\left(\\frac{f'-f}{\\sigma}\\right)^2

        """
        return np.array(data_set.get('weight')) * (resids / np.array(data_set.get('sigma'))) ** 2

    def _convert_parms_core(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """ Core conversion code using in both directions. Returns the appropriate min and max bounds. """
        lenx = len(x)
        if lenx == self.n_parms:
            min_, max_ = np.array(self.par_eng.active.range).T
        elif lenx == self.n_all_parms:
            min_, max_ = np.array(self.par_eng.range).T
        else:
            raise ValueError(f"Cannot parse x with length {lenx}. Must contain values for all parameters or values for"
                             f"active parameters.")

        return min_, max_


class ReaxFFError(BaseParamsError):
    """ ReaxFF error function. """

    @classmethod
    def from_classic_files(cls, path: Union[Path, str], **kwargs) -> 'ReaxFFError':
        """ Initializes the error function from classic ReaxFF files.

        Parameters
        ----------
        path
            Path to classic ReaxFF files, passed to :func:`.setup_reax_from_classic`.
        """
        dat_set, job_col, rxf_eng = setup_reax_from_classic(path)
        return cls(dat_set, job_col, rxf_eng, **kwargs)

    @classmethod
    def from_params_files(cls, path: Union[Path, str], **kwargs) -> 'ReaxFFError':
        """ Initializes the error function from ParAMS data files.

        Parameters
        ----------
        path
            Path to directory containing ParAMS data set, job collection and ReaxFF engine files (see
            :func:`.setup_reax_from_params`).
        """
        dat_set, job_col, rxf_eng = setup_reax_from_params(path)
        return cls(dat_set, job_col, rxf_eng, **kwargs)

    def toggle_parameter(self, parameters: Union[Sequence[int], Sequence[str]], toggle: Union[str, bool] = None,
                         force: bool = False):
        """ De/Activate parameters.
        This means either allowing them to be changed during an optimization, or fixing their value so that they are not
        changed.

        See :meth:`.toggle_parameter`.

        Parameters
        ----------
        force
            If :obj:`True`, the sense checks which verify that certain parameters are not activated will be bypassed.

        Warns
        -----
        UserWarning
            If `parameters` contains a parameter which should never be activated and `toggle` is :obj:`True` or
            :code:`'on'`.

        Notes
        -----
        Certain parameters should never be activated. For examples, some represent two- or three-way toggles for certain
        behaviours. Others can only take very specific values based on which atoms are present. This method will ignore
        and warn about attempts to activate such parameters unless `force` is used.
        """
        if toggle is True or toggle == 'on':
            allowed = np.array(self.par_eng._get_active())

            activating = np.full(self.n_all_parms, False)
            activating[[self.par_eng[i]._id for i in parameters]] = True

            invalid_act = np.argwhere((allowed == False) & (activating == True)).ravel()
            if invalid_act.size > 0:
                warnings.warn(f"The following parameters should never be activated: {invalid_act}.", UserWarning)

            valid_parameters = parameters if force else np.argwhere((allowed == True) & (activating == True)).ravel()
        else:
            valid_parameters = parameters
        super().toggle_parameter(valid_parameters, toggle)

    def checkpoint_save(self, path: Union[Path, str]):
        """ Used to store files into a GloMPO checkpoint (at path) suitable to reconstruct the task when the checkpoint
        is loaded.
        """
        path = Path(path).resolve(True)
        self.dat_set.pickle_dump(str(path / 'data_set.pkl'))
        self.job_col.pickle_dump(str(path / 'job_collection.pkl'))
        self.par_eng.pickle_dump(str(path / 'reax_params.pkl'))


class XTBError(BaseParamsError):
    """ GFN-xTB error function. """

    @classmethod
    def from_params_files(cls, path: Union[Path, str], **kwargs) -> 'XTBError':
        """ Initializes the error function from ParAMS data files.

        Parameters
        ----------
        path
            Path to directory containing ParAMS data set, job collection and ReaxFF engine files (see
            :func:`setup_reax_from_params`).
        """
        dat_set, job_col, rxf_eng = setup_xtb_from_params(path)
        return cls(dat_set, job_col, rxf_eng, **kwargs)

    def checkpoint_save(self, path: Union[Path, str]):
        """ Used to store files into a GloMPO checkpoint (at path) suitable to reconstruct the task when the checkpoint
        is loaded.
        """
        path = Path(path).resolve(True)
        self.dat_set.pickle_dump(str(path / 'data_set.pkl'))
        self.job_col.pickle_dump(str(path / 'job_collection.pkl'))
        self.par_eng.write(str(path))


def setup_reax_from_classic(path: Union[Path, str]) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """ Parses classic ReaxFF force field and configuration files into instances which can be evaluated by AMS.

    Parameters
    ----------
    path
        Path to directory containing classic ReaxFF configuration files:

    Notes
    -----
    `path` must contain:

        ``trainset.in``: Contains the description of the items in the training set.

        ``control``: Contains ReaxFF settings.

        ``geo``: Contains the geometries of the items used in the training set, will make the
        :class:`~scm.params.core.jobcollection.JobCollection` along with the ``control`` file.

        ``ffield``: A force field file which contains values for all the parameters. By default almost all parameters
        are activated and given ranges of :math:`\\pm 20%` if non-zero and [-1, 1] otherwise. See
        :class:`~scm.params.parameterinterfaces.reaxff.ReaxParams` for details.

    Optionally, the directory may contain:

        ``params``: File which describes which parameters to optimize and their ranges.

    Or, alternatively:

        ``ffield_bool``: A force field file with all parameters set to 0 or 1. 1 indicates it will be adjusted during
        optimization. 0 indicates it will not be changed during optimization.

        ``ffield_max``: A force field file where the active parameters are set to their maximum value (value of other
        parameters is ignored).

        ``ffield_min``: A force field file where the active parameters are set to their maximum value (value of other
        parameters is ignored).

    The method will ignore ``ffield_bool``, ``ffield_min`` and ``ffield_max`` if ``params`` is also present.

    .. caution::

       ``params`` files are not supported in ParAMS <v0.5.1. In this case the file will be ignored and the method will
       directly look for ``ffield_bool``, ``ffield_min`` and ``ffield_max``.

    Returns
    -------
    Tuple[~scm.params.core.dataset.DataSet, ~scm.params.core.jobcollection.JobCollection, ~scm.params.parameterinterfaces.reaxff.ReaxParams]
        ParAMS reparameterization objects: job collection, training set and engine.
    """

    path = Path(path).resolve(True)

    # Setup the dataset
    dat_set = trainset_to_params(str(path / 'trainset.in'))

    # Setup the job collection depending on the types of data in the training set
    settings = reaxff_control_to_settings(str(path / 'control'))
    if (PARAMS_VERSION_INFO == (0, 5, 0) and dat_set.forces()) or \
            (PARAMS_VERSION_INFO == (0, 5, 1) and dat_set.from_extractors('forces')):
        settings.input.ams.properties.gradients = True
    job_col = geo_to_params(str(path / 'geo'), settings)

    # Setup the Engine and parameters
    rxf_eng = ReaxParams(str(path / 'ffield'), bounds_scale=1.2)

    # Look for optional extras files
    params_path = path / 'params'
    bool_path = path / 'ffield_bool'
    min_path = path / 'ffield_min'
    max_path = path / 'ffield_max'

    if params_path.exists() and PARAMS_VERSION_INFO > (0, 5, 0):
        rxf_eng.read_paramsfile(str(params_path))
    elif all(extra.exists() for extra in (bool_path, min_path, max_path)):
        bool_eng = ReaxParams(str(path / 'ffield_bool'))
        max_eng = ReaxParams(str(path / 'ffield_max'))
        min_eng = ReaxParams(str(path / 'ffield_min'))

        rxf_eng.is_active = [bool(val) for val in bool_eng.x]

        for p in rxf_eng:
            if p.is_active:
                if min_eng[p.name].value < max_eng[p.name].value:
                    p.range = (min_eng[p.name].value, max_eng[p.name].value)
                else:
                    p.x = min_eng[p.name].value
                    p.is_active = False
                    print(f"WARNING: '{p.name}' deactivated due to bounds min >= max.")

    # Consistency Checks

    # Parameter value between bounds
    for p in rxf_eng.active:
        if not p.range[0] < p.value < p.range[1]:
            p.value = (p.range[0] + p.range[1]) / 2
            warnings.warn(f"'{p.name}' starting value out of bounds moving to midpoint.")

    # Remove training set entries not in job collection
    remove_ids = dat_set.check_consistency(job_col)
    if remove_ids:
        print('The following jobIDs are not in the JobCollection, their respective training set entries will be '
              'removed:')
        print('\n'.join({s for e in [dat_set[i] for i in remove_ids] for s in e.jobids}))
        del dat_set[remove_ids]

    return dat_set, job_col, rxf_eng


def _setup_collections_from_params(path: Union[Path, str]) -> Tuple[DataSet, JobCollection]:
    """ Loads ParAMS produced ReaxFF files into ParAMS objects.

    Parameters
    ----------
    path
        Path to folder containing:
            ``data_set.yml`` OR ``data_set.pkl``
                Contains the description of the items in the training set. A YAML file must be of the form produced by
                :meth:`~scm.params.core.dataset.DataSet.store`, a pickle file must be of the form produced by
                :meth:`~scm.params.core.dataset.DataSet.pickle_dump`. If both files are present, the pickle is given
                priority.
            ``job_collection.yml`` OR ``job_collection.pkl``
                Contains descriptions of the AMS jobs to evaluate. A YAML file must be of the form produced by
                :meth:`~scm.params.core.jobcollection.JobCollection.store`, a pickle file must be of the form produced
                by :meth:`scm.params.core.jobcollection.JobCollection.pickle_dump`.  If both files are present, the
                pickle is given priority.
    """
    dat_set = DataSet()
    job_col = JobCollection()

    for name, params_obj in {'data_set': dat_set, 'job_collection': job_col}.items():
        built = False
        for suffix, loader in {'.pkl': 'pickle_load', '.yml': 'load'}.items():
            file = Path(path, name + suffix)
            if file.exists():
                getattr(params_obj, loader)(str(file))
                built = True
        if not built:
            raise FileNotFoundError(f"No {name.replace('_', ' ')} data found")

    return dat_set, job_col


def setup_reax_from_params(path: Union[Path, str]) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """ Loads ParAMS produced ReaxFF files into ParAMS objects.

    Parameters
    ----------
    path
        Path to folder containing:
            ``data_set.yml`` OR ``data_set.pkl``
                Contains the description of the items in the training set. A YAML file must be of the form produced by
                :meth:`~scm.params.core.dataset.DataSet.store`, a pickle file must be of the form produced by
                :meth:`~scm.params.core.dataset.DataSet.pickle_dump`. If both files are present, the pickle is given
                priority.
            ``job_collection.yml`` OR ``job_collection.pkl``
                Contains descriptions of the AMS jobs to evaluate. A YAML file must be of the form produced by
                :meth:`~scm.params.core.jobcollection.JobCollection.store`, a pickle file must be of the form produced
                by :meth:`~scm.params.core.jobcollection.JobCollection.pickle_dump`.  If both files are present, the
                pickle is given priority.
            ``reax_params.pkl``
                Pickle produced by :meth:`~scm.params.parameterinterfaces.base.BaseParameters.pickle_dump`, representing
                the force field, active parameters and their ranges.
    """
    dat_set, job_col = _setup_collections_from_params(path)
    rxf_eng = ReaxParams.pickle_load(str(Path(path, 'reax_params.pkl')))

    return dat_set, job_col, rxf_eng


def setup_xtb_from_params(path: Union[Path, str]) -> Tuple[DataSet, JobCollection, XTBParams]:
    """ Loads ParAMS produced ReaxFF files into ParAMS objects.

    Parameters
    ----------
    path
        Path to folder containing:
            ``data_set.yml`` OR ``data_set.pkl``
                Contains the description of the items in the training set. A YAML file must be of the form produced by
                :meth:`~scm.params.core.dataset.DataSet.store`, a pickle file must be of the form produced by
                :meth:`~scm.params.core.dataset.DataSet.pickle_dump`. If both files are present, the pickle is given
                priority.
            ``job_collection.yml`` OR ``job_collection.pkl``
                Contains descriptions of the AMS jobs to evaluate. A YAML file must be of the form produced by
                :meth:`~scm.params.core.jobcollection.JobCollection.store`, a pickle file must be of the form produced
                by :meth:`~scm.params.core.jobcollection.JobCollection.pickle_dump`.  If both files are present, the
                pickle is given priority.
            ``elements.xtbpar``, ``basis.xtbpar``, ``globals.xtbpar``, ``additional_parameters.yaml``, ``metainfo.yaml``, ``atomic_configurations.xtbpar``, ``metals.xtbpar``
                Classic xTB parameter files.
    """
    dat_set, job_col = _setup_collections_from_params(path)
    xtb_eng = XTBParams(str(path))

    return dat_set, job_col, xtb_eng
