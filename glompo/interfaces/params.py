""" Provides support to use GloMPO with ParAMS.
    There are two ways to do this depending on your preferred workflow or interface.
    1) ParAMS is primary, setup an Optimization instance as normal.
       GloMPO is wrapped using the GlompoParamsWrapper below to look like a scm.params.optimizers.BaseOptimizer
    2) GloMPI is primary, setup a GloMPOManager instance as normal.
       The ReaxFFError class below will create the error function to be used as the manager 'task' parameter.
"""
import os
import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import dill
import numpy as np
from scm.params.common.parallellevels import ParallelLevels
from scm.params.common.reaxff_converter import geo_to_params, trainset_to_params
from scm.params.core.dataset import DataSet, Loss
from scm.params.core.jobcollection import JobCollection
from scm.params.core.lossfunctions import SSE
from scm.params.core.opt_components import LinearParameterScaler, _Step
from scm.params.optimizers.base import BaseOptimizer, MinimizeResult
from scm.params.parameterinterfaces.reaxff import ReaxParams
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

    def resids(self, pars):
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
                 function: Callable,
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

        manager = GloMPOManager(task=_FunctionWrapper(function), bounds=bounds, optimizer_selector=self.selector,
                                **self.manager_kwargs)

        result = manager.start_manager()

        # Reshape glompo.common.namedtuples.Result into scm.params.optimizers.base.MinimizeResult
        params_res = MinimizeResult()
        params_res.x = result.x
        params_res.fx = result.fx
        params_res.success = manager.converged and len(result.x) > 0

        return params_res


class ReaxFFError:
    """ Setups a function which when called returns the error value of a parameterised ReaxFF force field as compared to
        a provided trainign set of data.
    """

    def __init__(self, *,
                 path_to_classic: Optional[str] = None,
                 path_to_params: Optional[str] = None,
                 loss: Union[Loss, str, None] = None):
        """ Initialisation of the error function from configuration files.

            Parameters
            ----------
            path_to_classic: Optional[str] = None
                Path to classic ReaxFF files, passed to setup_reax_from_classic (see its docs for what files are
                expected).
            path_to_params: Optional[str] = None
                Path to directory containing ParAMS data set, job collection and ReaxFF engine files.
                (see setup_reax_from_params for what files are expected).
            loss: Union[Loss, str]
                A subclass of scm.params.core.dataset.Loss, holding the mathematical definition of the
                loss function to be applied to every entry, or a registered string shortcut.
        """
        assert not (path_to_classic and path_to_params), "Only one path allowed."

        if path_to_classic:
            self.dat_set, self.job_col, self.rxf_eng = setup_reax_from_classic(path_to_classic)
        elif path_to_params:
            self.dat_set, self.job_col, self.rxf_eng = setup_reax_from_params(path_to_params)
        else:
            self.dat_set, self.job_col, self.rxf_eng = (None,) * 3

        if loss:
            self.loss = loss
        else:
            self.loss = 'sse'
        self.scaler = LinearParameterScaler(self.rxf_eng.active.range)
        self.par_levels = ParallelLevels(jobs=1)

    @property
    def n_parms(self):
        """ Returns the number of active parameters. """
        return len(self.rxf_eng.active.x)

    def __call__(self, x: Sequence[float]):
        """ Returns the error value between the the force field with the given parameters and the training values. """
        return self._calculate(x)[0]

    def resids(self, x: Sequence[float]):
        """ Method for compatibility with GFLS optimizer. Returns the signed differences between the force field and
            training set.
        """
        return self._calculate(x)[1]

    def checkpoint_save(self, path: str):
        """ Used to store files into a GloMPO checkpoint (at path) suitable to reconstruct the task when the checkpoint
            is loaded.
        """
        self.dat_set.pickle_dump(os.path.join(path, 'data_set'))
        self.job_col.pickle_dump(os.path.join(path, 'job_collection'))
        with open(os.path.join(path, 'reax_engine'), 'wb') as file:
            dill.dump(self.rxf_eng, file)
        with open(os.path.join(path, 'extra_task'), 'wb') as file:
            dill.dump({'loss': self.loss,
                       'scaler': self.scaler,
                       'par_levels': self.par_levels}, file)

    def save(self, path: str, filenames: Optional[Dict[str, str]] = None, parameters: Optional[Sequence[float]] = None):
        """ Writes the data set and job collection to YAML files. Writes a ReaxFF force field file.

            Parameters
            ----------
            path: str
                Path to directory in which files will be saved.
            filenames: Optional[Dict[str, str]] = None
                Custom filenames for the written files. The dictionary may include any/all of the keys in the example
                below. This example contains the default names used if not given:
                    {'ds': 'data_set.yml', 'jc': 'job_collection.yml', 'ff': 'ffield'}
            parameters: Optional[Sequence[float]] = None
                Optional parameters to be written into the force field file. If not given the parameters currently,
                therein will be used.
        """
        names = {'ds': filenames['ds'] if 'ds' in filenames else 'data_set.yml',
                 'jc': filenames['jc'] if 'jc' in filenames else 'job_collection.yml',
                 'ff': filenames['ff'] if 'ff' in filenames else 'ffield'}

        self.dat_set.store(os.path.join(path, names['ds']))
        self.job_col.store(os.path.join(path, names['jc']))
        self.rxf_eng.write(os.path.join(path, names['ff']), parameters)

    def _calculate(self, x: Sequence[float]):
        """ Core calculation function, returns both the error function value and the residuals. """
        try:
            self.rxf_eng.active.x = self.scaler.scaled2real(x)
            engine = self.rxf_eng.get_engine()
            ff_results = self.job_col.run(engine.settings, parallel=self.par_levels)
            err_result = self.dat_set.evaluate(ff_results, self.loss, True)
            return err_result
        except ResultsError:
            return np.inf, np.array([np.inf])


def setup_reax_from_classic(path: str) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """
    Parses classic ReaxFF force field and configuration files into instances which can be evaluated by AMS.

    Parameters
    ----------
    path: str
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

    Returns
    -------
    DataSet, JobCollection, ReaxParams
        DataSet contains the training data
        JobCollection contains details of the PLAMS jobs which must be run to extract the force field results with which
            to compare to the the DataSet.
        ReaxParams is the interface to the actual engine used to calculate the force field results.
    """

    dat_set = trainset_to_params(os.path.join(path, 'trainset.in'))
    rxf_eng = ReaxParams(os.path.join(path, 'ffield_bool'))
    vars_max = ReaxParams(os.path.join(path, 'ffield_max'))
    vars_min = ReaxParams(os.path.join(path, 'ffield_min'))

    # Update the job collection depending on the types of data in the training set
    settings = reaxff_control_to_settings(os.path.join(path, 'control'))
    if dat_set.forces():
        settings.input.ams.properties.gradients = True
    job_col = geo_to_params(os.path.join(path, 'geo'), settings)

    # Remove training set entries not in job collection
    remove_ids = dat_set.check_consistency(job_col)
    if remove_ids:
        print(
            'The following jobIDs are not in the JobCollection, their respective training set entries will be removed:')
        print('\n'.join(set([s for e in [dat_set[i] for i in remove_ids] for s in e.jobids])))
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

    vars_values = ReaxParams(os.path.join(path, 'ffield_init'))
    rxf_eng.x = vars_values.x
    for parm in rxf_eng.active:
        if not parm.range[0] < parm.value < parm.range[1]:
            parm.value = (parm.range[0] + parm.range[1]) / 2
            warnings.warn("Starting value out of bounds moving to midpoint.")

    return dat_set, job_col, rxf_eng


def setup_reax_from_params(path: str) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """
    Loaded ParAMS produced ReaxFF files into ParAMS objects.

    Parameters
    ----------
    path: str
        Path to folder containing:
        - data_set.yml:       Contains the description of the items in the training set.
        - job_collection.yml: Contains descriptions of the AMS jobs to evaluate.
        - reax_params.pkl:    Pickle of the ReaxParams object which contains the force field, active variables and their
                              ranges.

    Returns
    -------
    DataSet, JobCollection, ReaxParams
        DataSet contains the training data
        JobCollection contains details of the PLAMS jobs which must be run to extract the force field results with which
            to compare to the the DataSet.
        ReaxParams is the interface to the actual engine used to calculate the force field results.
    """

    dat_set = DataSet().load(os.path.join(path, 'data_set.yml'))
    job_col = JobCollection().load(os.path.join(path, 'job_collection.yml'))
    with open(os.path.join(path, 'reax_params.pkl'), 'rb') as file:
        rxf_eng = dill.load(file)

    return dat_set, job_col, rxf_eng


def reaxfferror_task_loader(path: str) -> ReaxFFError:
    """ Builds a ReaxFFError instance from the files (located in path) made by its checkpoint_save method.
        This method was designed to be used as the task_loader when loading a GloMPO checkpoint.
    """
    task = ReaxFFError()

    dat_set = DataSet().pickle_load(os.path.join(path, 'data_set'))
    job_col = JobCollection().pickle_load(os.path.join(path, 'job_collection'))
    with open(os.path.join(path, 'reax_engine'), 'rb') as file:
        rxf_eng = dill.load(file)

    task.dat_set = dat_set
    task.job_col = job_col
    task.rxf_eng = rxf_eng

    return task
