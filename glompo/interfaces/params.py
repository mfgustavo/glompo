""" Provides support to use GloMPO with ParAMS.
    There are two ways to do this depending on your preferred workflow or interface.
    1) ParAMS is primary, setup an Optimization instance as normal.
       GloMPO is wrapped using the GlompoParamsWrapper below to look like a scm.params.optimizers.BaseOptimizer
    2) GloMPI is primary, setup a GloMPOManager instance as normal.
       The ReaxFFError class below will create the error function to be used as the manager 'task' parameter.
"""
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scm.params.common.parallellevels import ParallelLevels
from scm.params.common.reaxff_converter import geo_to_params, trainset_to_params
from scm.params.core.dataset import DataSet, SSE
from scm.params.core.jobcollection import JobCollection
from scm.params.core.opt_components import LinearParameterScaler, _Step
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
from scm.params.parameterinterfaces.base import BaseParameters
from scm.params.parameterinterfaces.reaxff import ReaxParams
from scm.params.parameterinterfaces.xtb import XTBParams
from scm.plams.core.errors import ResultsError
from scm.plams.interfaces.adfsuite.reaxff import reaxff_control_to_settings

from ..core.manager import GloMPOManager
from ..opt_selectors.baseselector import BaseSelector
from ..optimizers.gflswrapper import GFLSOptimizer

__all__ = ("GlompoParamsWrapper",
           "ReaxFFError")


class _FunctionWrapper:
    """ Wraps function produced by ParAMS internals (instance of scm.params.core.opt_components._Step) to match the API
        required by the 'task' parameter of GloMPOManager. Can be modified to achieve compatibility with
        other optimizers.

        Currently:
        1) Returns a float from the __call__ function;
        2) Add a resids parameter for compatibility with optsam GFLS algorithm.
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

    def resids(self, pars) -> Sequence[float]:
        """ Method added to conform the function to optsam API and allow the GFLS algorithm to be used. """

        result = self.func(pars, full=True)[0]

        resids = result.residuals
        dataset = result.dataset

        if len(resids) == 0:
            return np.array([np.inf])

        weights = dataset.get('weight')
        sigmas = dataset.get('sigma')

        resids = np.concatenate([(w / s) * r for w, s, r in zip(weights, sigmas, resids)])
        return resids


class GlompoParamsWrapper(BaseOptimizer):
    """ Wraps the GloMPO manager into a ParAMS optimizer. """

    def __init__(self, optimizer_selector: BaseSelector, **manager_kwargs):
        """ Accepts GloMPO configuration information.

            Parameters
            ----------
            optimizer_selector: BaseSelector
                Initialised BaseSelector object which specifies how optimizers are selected and initialised. See
                glompo.opt_selectors.BaseSelector for detailed documentation.
            **manager_kwargs
                Optional arguments to the GloMPOManager initialisation function.
                Note that all arguments are accepted but required GloMPO arguments 'task' and 'bounds'
                will be overwritten as they are passed by the 'minimize' function in accordance with ParAMS API.
        """

        self.manager_kwargs = manager_kwargs
        for kw in ['task', 'bounds']:
            if kw in self.manager_kwargs:
                del self.manager_kwargs[kw]

        self.selector = optimizer_selector

        if GFLSOptimizer in optimizer_selector:
            self._loss = SSE()

    def minimize(self,
                 function: _Step,
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 workers: int = 1) -> MinimizeResult:
        """
        Passes 'function' to GloMPO to be minimized. Returns and instance of MinimizeResult.

        Parameters
        ----------
        function: Callable
            Function to be minimized, this is passed as GloMPO's 'task' parameter.
        x0: Sequence[float]
            The length of this vector is taken to be the number of parameters in the optimization. It is not, however,
            used as the starting point for any optimizer the correct way to control this is by using GloMPO
            'BaseGenerator' objects.
        bounds: Sequence[Tuple[float, float]]
            Sequence of (min, max) pairs used to bound the search area for every parameter.
            The 'bounds' parameter is passed to GloMPO as its 'bounds' parameter.

            Note that by default ParAMS shifts and scales all parameters to the interval (0, 1). GloMPO will work in
            this space and be blind to the true bounds, thus results from the GloMPO logs cannot be applied directly
            to the function.
        workers: int
            Represents the maximum number of optimizers run in parallel. Passed to GloMPO as its 'max_jobs' parameter
            if 'max_jobs' has not been sent during initialisation via manager_kwargs otherwise ignored. If allowed to
            default this will usually result in the number of optimizers as there are cores available.

        Notes
        -----
        GloMPO is not currently compatible with using multiple DataSets and only the first one will be considered.

        Beware of using batching with GFLS as it requires all contributions to be evaluated every iteration.

        """

        warnings.warn("The x0 parameter is ignored by GloMPO. To control the starting locations of optimizers within "
                      "GloMPO make use of its BaseGenerator objects.", RuntimeWarning)

        if 'max_jobs' not in self.manager_kwargs:
            self.manager_kwargs['max_jobs'] = workers

        # Silence function printing
        function.v = False

        manager = GloMPOManager()
        manager.setup(task=_FunctionWrapper(function), bounds=bounds, opt_selector=self.selector,
                      **self.manager_kwargs)

        result = manager.start_manager()

        # Reshape glompo.common.namedtuples.Result into scm.params.optimizers.base.MinimizeResult
        params_res = MinimizeResult()
        params_res.x = result.x
        params_res.fx = result.fx
        params_res.success = manager.converged and len(result.x) > 0

        return params_res


class BaseParamsError:
    """ Base error function instance from which other classes derive depending on the engine used e.g. ReaxFF, xTB etc.
    """

    def __init__(self, data_set: DataSet, job_collection: JobCollection, parameters: BaseParameters,
                 validation_dataset: Optional[DataSet] = None,
                 scale_residuals: bool = True):
        """ Initialisation of the error function. To initialise the object from files use the class factory
            methods from_classic_files or from_params_files.

            Parameters
            ----------
            data_set: DataSet
                Reference data used to compare against force field results.
            job_collection: JobCollection
                AMS jobs from which the data can be extracted for comparison to the DataSet
            parameters: BaseParameters
                BaseParameters object which holds the force field values, ranges, engine and which parameters are active
                or not.
            validation_dataset: Optional[DataSet]
                If a validation set is being used and evaluated along with the training set, it may be added here.
                Jobs for the validation set are expected to be included in job_collection
            scale_residuals: bool = True
                If True then the raw residuals (i.e. the differences between engine evaluation and training data) will
                be scaled by the weight and sigma values in the datasets i.e. r_scaled = weight * r / sigma. Otherwise
                the raw residual is returned. This setting effects both the resids and detailed_call methods.
        """
        self.dat_set = data_set
        self.job_col = job_collection
        self.par_eng = parameters
        self.val_set = validation_dataset

        self.scale_residuals = scale_residuals

        self.loss = SSE()
        self.scaler = LinearParameterScaler(self.par_eng.active.range)
        self.par_levels = ParallelLevels(jobs=1)

    @property
    def n_parms(self) -> int:
        """ Returns the number of active parameters. """
        return len(self.par_eng.active.x)

    def __call__(self, x: Sequence[float]) -> float:
        """ Returns the error value between the the force field with the given parameters and the training values. """
        return self._calculate(x)[0][0]

    def detailed_call(self, x: Sequence[float]) -> Sequence[float]:
        """ A full return of the error results. Returns a sequence:
                [training_set_error, training_set_residual_1, ..., training_set_residual_N,
                 validation_set_error, validation_set_residual_1, ..., validation_set_residual_N]
            The list is truncated after the training set residuals if no validation set is present.
        """
        calc = self._calculate(x)
        ts_fx = [calc[0][0]]
        ts_resids = calc[0][1]
        vs_fx = []
        vs_resids = []

        if self.val_set:
            vs_fx = [calc[1][0]]
            vs_resids = calc[1][1]

        if self.scale_residuals:
            ts_resids = self._scale_residuals(ts_resids, self.dat_set)
            vs_resids = self._scale_residuals(vs_resids, self.val_set)

        return [*ts_fx, *ts_resids, *vs_fx, *vs_resids]

    def detailed_call_header(self) -> Sequence[str]:
        """ Returns a sequence of strings which represent the column headers for the detailed_call return.
            GloMPO optimizers will attached this sequence to the head of their CSV log files.
        """
        n_ts = len(self.dat_set)
        n_vs = len(self.val_set) if self.val_set else 0

        ts_digits = len(str(n_ts))
        vs_digits = len(str(n_vs))

        ts_heads = ['fx', *[f"r{i:0{ts_digits}}" for i in range(n_ts)]]
        vs_heads = ['fx', *[f"r{i:0{vs_digits}}_vs" for i in range(n_vs)]] if self.val_set else []

        return ts_heads + vs_heads

    def resids(self, x: Sequence[float]) -> Sequence[float]:
        """ Method for compatibility with GFLS optimizer. Returns the signed differences between the force field and
            training set but DOES NOT include weights and sigma values
        """
        residuals = self._calculate(x)[0][1]
        if self.scale_residuals:
            residuals = self._scale_residuals(residuals, self.dat_set)

        return residuals

    def save(self, path: Union[Path, str], filenames: Optional[Dict[str, str]] = None,
             parameters: Optional[Sequence[float]] = None):
        """ Writes the data set and job collection to YAML files. Writes the engine object to an appropriate parameter
            file.

            Parameters
            ----------
            path: Union[Path, str]
                Path to directory in which files will be saved.
            filenames: Optional[Dict[str, str]] = None
                Custom filenames for the written files. The dictionary may include any/all of the keys in the example
                below. This example contains the default names used if not given:
                    {'ds': 'data_set.yml', 'jc': 'job_collection.yml', 'ff': 'ffield'}
            parameters: Optional[Sequence[float]] = None
                Optional parameters to be written into the force field file. If not given, the parameters currently
                therein will be used.
        """
        if not filenames:
            filenames = {}

        names = {'ds': filenames['ds'] if 'ds' in filenames else 'data_set.yml',
                 'jc': filenames['jc'] if 'jc' in filenames else 'job_collection.yml',
                 'ff': filenames['ff'] if 'ff' in filenames else 'ffield'}

        self.dat_set.store(Path(path, names['ds']))
        self.job_col.store(Path(path, names['jc']))
        self.par_eng.write(Path(path, names['ff']), parameters)

    def _calculate(self, x: Sequence[float]) -> Sequence[Tuple[float, List[float], List[float]]]:
        """ Core calculation function, returns both the error function value and the residuals. """
        default = (float('inf'), [float('inf')], [float('inf')])
        try:
            engine = self.par_eng.get_engine(self.scaler.scaled2real(x))
            ff_results = self.job_col.run(engine.settings, parallel=self.par_levels)
            ts_result = self.dat_set.evaluate(ff_results, self.loss, True)
            vs_result = self.val_set.evaluate(ff_results, self.loss, True) if self.val_set else default
            return ts_result, vs_result
        except ResultsError:
            return default, default

    @staticmethod
    def _scale_residuals(resids: Sequence[float], data_set: DataSet) -> Sequence[float]:
        """ Scales a sequence of residuals by weight and sigma values in the associated DataSet"""
        return [w / s * r for w, s, r in zip(data_set.get('weight'), data_set.get('sigma'), resids)]


class ReaxFFError(BaseParamsError):
    """ Setups a function which when called returns the error value of a parameterised ReaxFF force field as compared to
        a provided training set of data.
    """

    @classmethod
    def from_classic_files(cls, path: Union[Path, str]) -> 'ReaxFFError':
        """ Initializes the error function from classic ReaxFF files.

            Parameters
            ----------
            path: Union[Path, str]
                Path to classic ReaxFF files, passed to setup_reax_from_classic (see its docs for what files are
                expected).
        """
        dat_set, job_col, rxf_eng = setup_reax_from_classic(path)
        return cls(dat_set, job_col, rxf_eng)

    @classmethod
    def from_params_files(cls, path: Union[Path, str]) -> 'ReaxFFError':
        """ Initializes the error function from ParAMS data files.

            Parameters
            ----------
            path: Union[Path, str]
                Path to directory containing ParAMS data set, job collection and ReaxFF engine files.
                (see setup_reax_from_params for what files are expected).
        """
        dat_set, job_col, rxf_eng = setup_reax_from_params(path)
        return cls(dat_set, job_col, rxf_eng)

    def checkpoint_save(self, path: Union[Path, str]):
        """ Used to store files into a GloMPO checkpoint (at path) suitable to reconstruct the task when the checkpoint
            is loaded.
        """
        self.dat_set.pickle_dump(Path(path, 'data_set.pkl'))
        self.job_col.pickle_dump(Path(path, 'job_collection.pkl'))
        self.par_eng.pickle_dump(str(Path(path, 'reax_params.pkl')))  # Method does not support Path


class XTBError(BaseParamsError):
    """ Setups a function which when called returns the error value of a parameterised xTB force field as compared to
        a provided training set of data.
    """

    @classmethod
    def from_params_files(cls, path: Union[Path, str]) -> 'XTBError':
        """ Initializes the error function from ParAMS data files.

            Parameters
            ----------
            path: Union[Path, str]
                Path to directory containing ParAMS data set, job collection and ReaxFF engine files.
                (see setup_reax_from_params for what files are expected).
        """
        dat_set, job_col, rxf_eng = setup_xtb_from_params(path)
        return cls(dat_set, job_col, rxf_eng)

    def checkpoint_save(self, path: Union[Path, str]):
        """ Used to store files into a GloMPO checkpoint (at path) suitable to reconstruct the task when the checkpoint
            is loaded.
        """
        self.dat_set.pickle_dump(Path(path, 'data_set.pkl'))
        self.job_col.pickle_dump(Path(path, 'job_collection.pkl'))
        self.par_eng.write(path)


def setup_reax_from_classic(path: Union[Path, str]) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """
    Parses classic ReaxFF force field and configuration files into instances which can be evaluated by AMS.

    Parameters
    ----------
    path: Union[Path, str]
        Path to folder containing:
        - trainset.in: Contains the description of the items in the training set
        - control:     Contains ReaxFF settings
        - ffield_init: A force field file which contains values for all the parameters
        - ffield_bool: A force field file with all parameters set to 0 or 1.
                       1 indicates it will be adjusted during optimisation.
                       0 indicates it will not be changed during optimisation.
        - ffield_max:  A force field file where the active parameters are set to their maximum value (value of other
                       parameters is ignored).
        - ffield_min:  A force field file where the active parameters are set to their maximum value (value of other
                       parameters is ignored).
        - geo:         Contains the geometries of the items used in the training set.
    """

    dat_set = trainset_to_params(Path(path, 'trainset.in'))
    rxf_eng = ReaxParams(Path(path, 'ffield_bool'))
    vars_max = ReaxParams(Path(path, 'ffield_max'))
    vars_min = ReaxParams(Path(path, 'ffield_min'))

    # Update the job collection depending on the types of data in the training set
    settings = reaxff_control_to_settings(Path(path, 'control'))
    if dat_set.forces():
        settings.input.ams.properties.gradients = True
    job_col = geo_to_params(Path(path, 'geo'), settings)

    # Remove training set entries not in job collection
    remove_ids = dat_set.check_consistency(job_col)
    if remove_ids:
        print(
            'The following jobIDs are not in the JobCollection, their respective training set entries will be removed:')
        print('\n'.join({s for e in [dat_set[i] for i in remove_ids] for s in e.jobids}))
        del dat_set[remove_ids]

    rxf_eng.is_active = [bool(val) for val in rxf_eng.x]

    for i, parm in enumerate(rxf_eng):
        if parm.is_active:
            if vars_min[i].value != vars_max[i].value:
                parm.range = (vars_min[i].value, vars_max[i].value)
            else:
                parm.x = vars_min[i].value
                parm.is_active = False
                print(f"WARNING: {parm.name} deactivated due to bounds.")

    vars_values = ReaxParams(Path(path, 'ffield_init'))
    rxf_eng.x = vars_values.x
    for parm in rxf_eng.active:
        if not parm.range[0] < parm.value < parm.range[1]:
            parm.value = (parm.range[0] + parm.range[1]) / 2
            warnings.warn("Starting value out of bounds moving to midpoint.")

    return dat_set, job_col, rxf_eng


def _setup_collections_from_params(path: Union[Path, str]) -> Tuple[DataSet, JobCollection]:
    """ Loads ParAMS produced ReaxFF files into ParAMS objects.

        Parameters
        ----------
        path: Union[Path, str]
            Path to folder containing:
            - data_set.yml OR data_set.pkl
                Contains the description of the items in the training set. A YAML file must be of the form produced by
                scm.params.core.dataset.DataSet.store, a pickle file must be of the form produced by
                scm.params.core.dataset.DataSet.pickle_dump. If both files are present, the pickle is given priority.
            - job_collection.yml OR job_collection.pkl
                Contains descriptions of the AMS jobs to evaluate. A YAML file must be of the form produced by
                scm.params.core.jobcollection.JobCollection.store, a pickle file must be of the form produced by
                scm.params.core.jobcollection.JobCollection.pickle_dump.  If both files are present, the pickle is given
                priority.
    """
    dat_set = DataSet()
    job_col = JobCollection()

    for name, params_obj in {'data_set': dat_set, 'job_collection': job_col}.items():
        built = False
        for suffix, loader in {'.pkl': 'pickle_load', '.yml': 'load'}.items():
            file = Path(path, name + suffix)
            if file.exists():
                getattr(params_obj, loader)(file)
                built = True
        if not built:
            raise FileNotFoundError(f"No {name.replace('_', ' ')} data found")

    return dat_set, job_col


def setup_reax_from_params(path: Union[Path, str]) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """ Loads ParAMS produced ReaxFF files into ParAMS objects.

        Parameters
        ----------
        path: Union[Path, str]
            Path to folder containing:
            - data_set.yml OR data_set.pkl
                Contains the description of the items in the training set. A YAML file must be of the form produced by
                scm.params.core.dataset.DataSet.store, a pickle file must be of the form produced by
                scm.params.core.dataset.DataSet.pickle_dump. If both files are present, the pickle is given priority.
            - job_collection.yml OR job_collection.pkl
                Contains descriptions of the AMS jobs to evaluate. A YAML file must be of the form produced by
                scm.params.core.jobcollection.JobCollection.store, a pickle file must be of the form produced by
                scm.params.core.jobcollection.JobCollection.pickle_dump.  If both files are present, the pickle is given
                priority.
            - reax_params.pkl:
                Pickle produced by scm.params.parameterinterfaces.reaxff.ReaxParams.pickle_dump, representing the force
                field, active parameters and their ranges.
    """
    dat_set, job_col = _setup_collections_from_params(path)
    rxf_eng = ReaxParams.pickle_load(Path(path, 'reax_params.pkl'))

    return dat_set, job_col, rxf_eng


def setup_xtb_from_params(path: Union[Path, str]) -> Tuple[DataSet, JobCollection, XTBParams]:
    """ Loads ParAMS produced ReaxFF files into ParAMS objects.

        Parameters
        ----------
        path: Union[Path, str]
            Path to folder containing:
            - data_set.yml OR data_set.pkl
                Contains the description of the items in the training set. A YAML file must be of the form produced by
                scm.params.core.dataset.DataSet.store, a pickle file must be of the form produced by
                scm.params.core.dataset.DataSet.pickle_dump. If both files are present, the pickle is given priority.
            - job_collection.yml OR job_collection.pkl
                Contains descriptions of the AMS jobs to evaluate. A YAML file must be of the form produced by
                scm.params.core.jobcollection.JobCollection.store, a pickle file must be of the form produced by
                scm.params.core.jobcollection.JobCollection.pickle_dump.  If both files are present, the pickle is given
                priority.
            - elements.xtbpar, basis.xtbpar, globals.xtbpar, additional_parameters.yaml, metainfo.yaml,
              atomic_configurations.xtbpar, metals.xtbpar
                Classic xTB parameter files.
    """
    dat_set, job_col = _setup_collections_from_params(path)
    xtb_eng = XTBParams(path)

    return dat_set, job_col, xtb_eng
