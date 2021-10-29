import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import numpy as np
import psutil
import tables as tb
from scm.params.common.helpers import plams_initsettings, printnow, strpad
from scm.params.core.callbacks import Callback
from scm.params.core.dataset import DataSet
from scm.params.core.jobcollection import JobCollection
from scm.params.core.lossfunctions import LOSSDICT, Loss
from scm.params.core.opt_components import EvaluatorReturn, LinearParameterScaler, _LossEvaluator, _Step
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
from scm.params.parameterinterfaces.base import BaseParameters, Constraint
from scm.plams.core.functions import config, finish, init
from scm.plams.core.jobrunner import JobRunner

from ...convergence.nconv import NOptConverged
from ...core.manager import GloMPOManager
from ...generators.random import RandomGenerator
from ...opt_selectors.cycle import CycleSelector
from ...optimizers.cmawrapper import CMAOptimizer


class _GloMPOEvaluatorReturn(NamedTuple):
    fx: float
    residuals: np.ndarray
    time: float


class ParallelLevels:
    """Specification of how to use parallelism at the different levels involved in a ParAMS optimization.

    Note that the parallel levels are layered in the order of the arguments
    to this method, e.g. setting `parametervectors=2` and `jobs=4` would
    result in two parameter vectors tried in parallel, with the |JC| for
    each one being run with 4 jobs in parallel, resulting in a total of 8
    jobs running concurrently. Overall this means that for fully using a
    machine the product of the parallelism at all levels should equal the
    number of (physical) CPU cores of that machine. It is generally more
    efficient to parallelize at a higher level, especially when
    parameterizing a very fast engine like ReaxFF.

    All parameters of this method are optional. Parameters not specified
    will be assigned a sensible default value resulting in the use of the
    entire machine.

    :Parameters:

        optimizations: optional, int > 0
            How many independent optimizations to run in parallel. This is
            a placeholder for the future. ParAMS currently does not support
            optimization level parallelism yet.
        parametervectors: optional, int > 0
            How many parameter vectors to try in parallel. This level of
            parallelism can only be used with parallel ref:`optimizers
            <Optimizers>`.
        jobs: optional, int > 0
            How many jobs from the |JC| to run in parallel.
        processes: optional, int
            How many processes (MPI ranks) to spawn for each job. This
            effectively sets the NSCM environment variable for each job.
            A value of `-1` will disable explicit setting of related variables.
        threads: optional, int
            How many threads to use for each of the processes. This
            effectively set the OMP_NUM_THREADS environment variable. Note
            that most AMS engines do not use threads, so the value of this
            variable would not have any effect. We recommend leaving it at
            the default value of ``1``. Please consult the manual of the
            engine you are parameterizing.
            A value of `-1` will disable explicit setting of related variables.

    :Attributes:

        workers: `int`
            The product of all parameters
    """

    def __init__(self, optimizations: int = 1, parametervectors: int = None, jobs: int = None, processes: int = 1,
                 threads: int = 1):
        self.optimizations = optimizations
        self.parametervectors = parametervectors
        self.jobs = jobs
        self.processes = processes
        self.threads = threads

        # Reasonable defaults for parallelization at levels not set by the user.
        num_cores = psutil.cpu_count(logical=False)  # Total number of *physical* CPU cores in this machine.
        if (self.parametervectors is None) and (self.jobs is None):
            # Parallelization at the parameter vector level is likely more efficient.
            self.parametervectors = int(
                num_cores / (self.optimizations * max(1, self.processes) * max(1, self.threads)))
            self.jobs = 1
        elif self.parametervectors is None:
            self.parametervectors = max(
                int(num_cores / (self.optimizations * self.jobs * max(1, self.processes) * max(1, self.threads))), 1)
        elif self.jobs is None:
            self.jobs = max(int(num_cores / (
                        self.optimizations * self.parametervectors * max(1, self.processes) * max(1, self.threads))), 1)

        # Make sure all are ints
        for i in ['optimizations', 'parametervectors', 'jobs', 'processes', 'threads']:
            v = getattr(self, i)
            setattr(self, i, int(v))

    def __str__(self):
        args = ', '.join(
            f"{i}={getattr(self, i)}" for i in ['optimizations', 'parametervectors', 'jobs', 'processes', 'threads'])
        return f"{self.__class__.__name__}({args})"

    def __repr__(self):
        return str(self)

    def copy(self):
        return self.__class__(
            optimizations=self.optimizations,
            parametervectors=self.parametervectors,
            jobs=self.jobs,
            processes=self.processes,
            threads=self.threads
        )

    @property
    def workers(self):
        return self.optimizations * self.parametervectors * self.jobs * max(1, self.processes) * max(1, self.threads)


class _GloMPOStep(_Step):
    def __init__(self, *args, workers, **kwargs):
        super().__init__(*args, **kwargs)
        self.workers = workers  # Bury inside _Step so workers arg not needed in call

    def __call__(self, x: Sequence[float], workers=1, full=False, _force=False) -> Union[float, _GloMPOEvaluatorReturn]:
        """ GloMPO cannot handle a full EvaluatorReturn """
        # todo cannot handle multiple x values. need to change in glompo wrapper.
        call = super().__call__(x, workers, full, _force)

        if not full:
            return call

        call: List[EvaluatorReturn]
        ret = tuple(i for ev in call for i in (ev.fx, ev.residuals, ev.time))
        return ret

    def detailed_call(self, x: Sequence) -> Sequence[float]:
        return self(x, self.workers, True, False)

    def headers(self) -> Dict[str, tb.Col]:
        heads = {}
        for i, loss_eval in enumerate(self.datasets):
            heads[loss_eval.name + '_fx'] = tb.Float64Col(pos=3 * i + 0)
            heads[loss_eval.name + '_residuals'] = tb.Float64Col((1, len(loss_eval.dataset)), pos=3 * i + 1)
            heads[loss_eval.name + '_time'] = tb.Float16Col(pos=3 * i + 2)

        return heads


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
        non-finite loss is probably due to bad :class:`.plams.Settings` of an entry in the `jobcollection``. However,
        if it is not known if the initial parameters can be trusted or raising an error is not desired for other
        reasons, this parameter can be set to :obj:`True` to skip the initial evaluation.
    logger_every
        See `every_n_iter` in :class:`~scm.params.core.callbacks.Logger`. This option is ignored if a Logger is provided
        in the callbacks.

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
        self.result = None

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
        def checkloss(loss_, pre=''):
            if isinstance(loss_, str):
                assert loss_.lower() in LOSSDICT, f"The requested loss '{loss_}' is not known."
                loss_ = LOSSDICT[loss_.lower()]()
            assert isinstance(loss_, Loss), f"{pre} Loss argument must be a subclass of the `Loss` class."
            return loss_

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
        self.skip_x0 = skip_x0
        # todo implement verbose
        self.verbose = verbose

        self.working_dir = Path(title)
        if self.working_dir.exists():
            i = 1
            while self.working_dir.with_suffix(f".{i:03}").exists():
                i += 1
            print(
                f"'{self.working_dir}' already exists. Will use '{self.working_dir.with_suffix(f'.{i:03}')}' instead.")
            self.working_dir = self.working_dir.with_suffix(f'.{i:03}')

        self.glompo = GloMPOManager()
        # todo discuss and review
        glompo_default_config = {'opt_selector': CycleSelector((CMAOptimizer,
                                                                {'workers': self.parallel.workers //
                                                                            self.parallel.optimizations},
                                                                {'sigma0': 0.5})),
                                 'convergence_checker': NOptConverged(self.parallel.optimizations),
                                 'x0_generator': RandomGenerator(self.scaler.bounds),
                                 'killing_conditions': None,
                                 'share_best_solutions': False,
                                 'hunt_frequency': 999999999,
                                 'checkpoint_control': None,
                                 'summary_files': 3,
                                 'is_log_detailed': True,
                                 'visualisation': False,
                                 'visualisation_args': None,
                                 'force_terminations_after': -1,
                                 'aggressive_kill': False,
                                 'end_timeout': None,
                                 'split_printstreams': True}
        for ignore in ('bounds', 'task', 'working_dir', 'overwrite_existing', 'max_jobs', 'backend'):
            if ignore in glompo_kwargs:
                del glompo_kwargs[ignore]
        self.glompo_kwargs = {**glompo_default_config, **glompo_kwargs}

    def optimize(self) -> MinimizeResult:
        """ Start the optimization given the initial parameters. """
        printnow(f'Starting parameter optimization. Dim = {len(self.interface.active)}')

        self.working_dir.mkdir(parents=True, exist_ok=False)

        sum_file = self.working_dir / 'summary.txt'
        self.summary(file=sum_file)
        with sum_file.open('a') as f:
            print(f"Start time: {datetime.now()}", file=f)

        idir = self.working_dir / 'settings_and_initial_data'
        idir.mkdir(parents=True, exist_ok=True)

        self.interface.pickle_dump(str(idir / 'initial_parameter_interface.pkl'))
        self.jobcollection.store(str(idir / 'jobcollection.yaml.gz'))

        (idir / 'datasets').mkdir(parents=True, exist_ok=True)
        for obj in self.objective:
            obj.dataset.store(str(idir / 'datasets' / obj.name) + '.yaml.gz')

        init(config_settings=plams_initsettings, path=self.plams_workdir_path)
        # All parallel parameter vectors share the same job runner, so the total number of jobs we want to have
        # running is the product of the parallelism at both optimizer and job collection level! (Like this we get
        # dynamic load balancing between different parameter vectors for free.)
        # todo change parallellevels docs to show that optimizations is now used.
        config.default_jobrunner = JobRunner(parallel=True,
                                             maxjobs=self.parallel.optimizations *
                                                     self.parallel.parametervectors *
                                                     self.parallel.jobs)

        # Single evaluation with the initial parameters
        fx0 = float('inf')
        if not self.skip_x0:
            fx0 = self.initial_eval()
        for i in self.objective:
            i.constraints = self.constraints  # do not include constraints in initial evaluation

        x0 = self.scaler.real2scaled(self.interface.active.x)
        f = _GloMPOStep(self.objective, None,
                        workers=self.parallel.parametervectors, verbose=self.verbose)  # make one callable function

        self.glompo.setup(task=f,
                          bounds=self.scaler.bounds,
                          working_dir=self.working_dir,
                          overwrite_existing=False,
                          max_jobs=self.parallel.workers,
                          backend='threads',
                          **self.glompo_kwargs)

        # Optimization
        mr = MinimizeResult()

        try:
            result = self.glompo.start_manager()

            mr.success = True
            mr.x = self.scaler.scaled2real(result.x)
            mr.fx = result.fx

            # Add initial eval to log
            g_log = self.working_dir / 'glompo_log.h5'
            if g_log.exists():
                with tb.open_file(g_log, 'a') as file:
                    file.root._v_attrs.initial_parameter_results = {'x': x0, 'fx': fx0}
        except Exception:
            traceback.print_exc()
            mr.success = False
        finally:
            self.result = mr

        finish()

        self.interface.active.x = self.result.x

        with sum_file.open('a') as f:
            print(f"End time:   {datetime.now()}", file=f)

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

        printnow(f"Initial loss: {fx:.3e}")

        return fx

    def summary(self, file=None):
        """ Prints a summary of the current instance """
        if file:
            with Path(file).open('w') as f:
                print(self, file=f)
        else:
            print(self)

    def __str__(self):
        s = f"{self.__class__.__name__}() Instance Settings:\n"
        l = len(s) - 1
        s += l * '=' + '\n'
        s += strpad(os.path.abspath(self.working_dir), 35, "Workdir:")
        s += strpad(len(self.jobcollection), 35, "JobCollection size:")
        s += strpad(self.interface.__class__.__name__, 35, "Interface:")
        s += strpad(len(self.interface.active), 35, "Active parameters:")
        # todo fix optimizer info
        s += strpad('GloMPO', 35, "Optimizer:")
        s += strpad(self.parallel, 35, "Parallelism:")
        s += strpad(self.verbose, 35, "Verbose:")
        if self.constraints:
            s += strpad(repr(self.constraints[0]), 35, "Constraints:")
            if len(self.constraints) > 1:
                for cb in self.constraints[1:]:
                    s += strpad(repr(cb), 35)
        s += strpad(self.plams_workdir_path, 35, "PLAMS workdir path:")
        s += '\nEvaluators:\n'
        s += '-----------\n'
        s += ''.join(str(i) for i in self.objective)
        s += '==='
        return s

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
                assert validation < len(datasets[0]), \
                    f"Requested validation set {validation} is larger than the training set {len(datasets[0])}"
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
        for num, (_ds, _loss, _batch_size, _use_pipe, _name, _eval_every, _maxjobs, _maxjobs_shuffle, _use_pipe) in \
                enumerate(zip(datasets, loss, batch_size, use_pipe, dataset_names, eval_every, maxjobs, maxjobs_shuffle,
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
            assert all(i is not None for i in _ds.get('reference')), \
                f"\n\nNot all entries in {_name} have a reference value. Please set or calculate it before starting " \
                f"the optimization.\nSee https://www.scm.com/doc/params/components/dataset/dataset.html for help.\n"
            objective.append(evaluator)

        return objective
