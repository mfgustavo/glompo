import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import numpy as np
import tables as tb
from scm.params.common.helpers import plams_initsettings, printerr, printnow
from scm.params.common.parallellevels import ParallelLevels
from scm.params.core.callbacks import Callback
from scm.params.core.dataset import DataSet
from scm.params.core.jobcollection import JobCollection
from scm.params.core.lossfunctions import LOSSDICT, Loss
from scm.params.core.opt_components import EvaluatorReturn, SCALER_DICT, _Step
from scm.params.core.parameteroptimization import Optimization
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
from scm.params.parameterinterfaces.base import BaseParameters, Constraint
from scm.plams.core.functions import config, finish, init
from scm.plams.core.jobrunner import JobRunner

from ...convergence.nconv import NOptConverged
from ...core.manager import GloMPOManager
from ...generators.random import RandomGenerator
from ...opt_selectors.cycle import CycleSelector
from ...opt_selectors.spawncontrol import NOptimizersSpawnStop
from ...optimizers.cmawrapper import CMAOptimizer


class _GloMPOEvaluatorReturn(NamedTuple):
    fx: float
    residuals: np.ndarray
    time: float


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
        ret = tuple(i for ev in call for i in (ev.fx, ev.time))
        return ret

    def detailed_call(self, x: Sequence) -> Sequence[float]:
        return self(x, self.workers, True, False)

    def headers(self) -> Dict[str, tb.Col]:
        heads = {}
        for i, loss_eval in enumerate(self.datasets):
            heads[loss_eval.name + '_time'] = tb.Float16Col(pos=i + len(self.datasets))

        return heads


class Optimization(Optimization):
    """
    Subclass of :class:`~.scm.params.core.parameteroptimization.Optimization`. While the parent is liable to change,
    the child overrides as little as possible. This should change later to make the code cleaner.

    For compatibility the signature remains the same.

    For most parameters the meaning of the parameters remains the same, only those that have changed are documented
    below.

    Parameters
    ----------
    optimizer
        IGNORED.

    title : optional, str
        The working directory for this optimization. Once :meth:`optimize` is called, will *NOT* switch to it.
        (see `glompo_kwargs`)

    callbacks
        IGNORED.

    verbose
        Active GloMPO's logging progress feedback.

    logger_every
        IGNORED.

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

        assert issubclass(parameterinterface.__class__,
                          BaseParameters), f"Parameter interface type not understood: {type(parameterinterface)}. Must be a subclass of BaseParameters."
        self.interface = parameterinterface
        assert len(
            self.interface.active) > 0, "The parameter interface does not contain any parameters marked for optimization. Check the `interface.is_active` attribute and make sure that at least one parameter is marked as active."
        for p in self.interface.active:
            assert p.range[0] <= p.value <= p.range[
                1], f"The ranges or value of {repr(p)} are ill-defined. Please make sure that `range[0] <= value <= range[1]`."

        # Warnings about GloMPO / ParAMS interface differences
        if callbacks:
            print("WARNING: Callbacks are ignored.")
        print("WARNING: The 'optimizer' argument is ignored. Please send optimizer related arguments to glompo_kwargs")
        self.callbacks = []

        # Scaler must be GloMPO-wide.
        # Enforce [0,1] scaling by default, other scalers can be requested in glompo_kwargs['scaler']
        scaler = SCALER_DICT[glompo_kwargs['scaler']] if 'scaler' in glompo_kwargs else SCALER_DICT['linear']
        self.scaler = scaler(self.interface.active.range)

        # Check loss
        def checkloss(loss, pre=''):
            if isinstance(loss, str):
                assert loss.lower() in LOSSDICT, f"The requested loss '{loss}' is not known."
                loss = LOSSDICT[loss.lower()]()
            assert isinstance(loss, Loss), f"{pre} Loss argument must be a subclass of the `Loss` class."
            return loss

        loss = [checkloss(i) for i in loss] if isinstance(loss, List) else checkloss(loss)

        if constraints is not None:
            assert all(isinstance(i, (Constraint, Constraint._Operator)) for i in
                       constraints), f"Unexpected constraints class provided: {', '.join([str(i.__class__) for i in constraints])}."
            if not all(c(self.interface) for c in constraints):
                printnow(
                    "WARNING: Initial interface violates constraints! This might result in undesired optimizer behaviour.")
        self.constraints = constraints

        if parallel is None:
            self.parallel = ParallelLevels()
        else:
            assert isinstance(parallel,
                              ParallelLevels), f"Type of the parallel argument is{type(parameterinterface)}, but should be ParallelLevels."
            self.parallel = parallel

        # Sets self.objective
        self._wrap_datasets(datasets, validation, loss, batch_size, use_pipe, dataset_names, eval_every, maxjobs,
                            maxjobs_shuffle)

        self.plams_workdir_path = plams_workdir_path or os.getenv('SCM_TMPDIR', '/tmp')
        self.skip_x0 = skip_x0

        self.verbose = verbose
        if verbose:
            formatter = logging.Formatter("%(asctime)s :: %(message)s")

            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            handler.setLevel('INFO')

            logger = logging.getLogger('glompo.manager')
            logger.setLevel('INFO')
            logger.addHandler(handler)

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
                                                                {'sigma0': 0.5}),
                                                               allow_spawn=NOptimizersSpawnStop(
                                                                   self.parallel.optimizations)),
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

        # The following are only set for compatibility with methods that we choose not to override for now.
        self.title = str(self.working_dir)
        self.optimizer = self.glompo

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
        # Unclear why verbose is sent to _Step???
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
            printnow(f"Optimization done.")

            # Add initial eval to log
            g_log = self.working_dir / 'glompo_log.h5'
            if g_log.exists():
                with tb.open_file(g_log, 'a') as file:
                    file.root._v_attrs.initial_parameter_results = {'x': x0, 'fx': fx0}
        except Exception:
            traceback.print_exc()
            mr.success = False
            printerr(f"WARNING: Optimization unsuccessful.")
        finally:
            self.result = mr

        finish()

        self.interface.active.x = self.result.x

        with sum_file.open('a') as f:
            print(f"End time:   {datetime.now()}", file=f)

        printnow(f"Final loss: {self.result.fx:.3e}")

        return self.result
