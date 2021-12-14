import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tables as tb
from scm.params.common._version import __version__
from scm.params.common.parallellevels import ParallelLevels
from scm.params.core.dataset import DataSet
from scm.params.core.jobcollection import JobCollection
from scm.params.core.lossfunctions import SSE
from scm.params.parameterinterfaces.base import BaseParameters
from scm.plams.core.errors import ResultsError

from .paramsbuilders import setup_reax_from_classic, setup_reax_from_params, setup_xtb_from_params

try:
    from scm.params.core.dataset import DataSetEvaluationError
except ImportError:
    # Different versions of ParAMSs raise different error types.
    DataSetEvaluationError = ResultsError

PARAMS_VERSION_INFO = tuple(map(int, __version__.split('.')))


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

    loss : Union[str, ~scm.params.core.lossfunctions.Loss]
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
        :attr:`active_names`
        :meth:`convert_indices_abs2rel`
        :meth:`convert_indices_rel2abs`
        """
        return [p._id for p in self.par_eng.active]

    @property
    def active_names(self) -> List[str]:
        """ Returns the names of the active parameters.

        See Also
        --------
        :attr:`active_abs_indices`
        :meth:`convert_indices_abs2rel`
        :meth:`convert_indices_rel2abs`
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
        heads = {'resids_ts': tb.Float64Col((1, len(self.dat_set)), pos=0)}

        if self.val_set:
            heads['fx_vs'] = tb.Float64Col(pos=1)
            heads['resids_vs'] = tb.Float64Col((1, len(self.val_set)), pos=2)

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

        >>> err.convert_indices_abs2rel([23, 57, 1])
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
        :attr:`convert_indices_rel2abs`
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
    def from_classic_files(cls,
                           path: Union[Path, str],
                           validation_dataset: Optional[DataSet] = None,
                           scale_residuals: bool = False,
                           **kwargs) -> 'ReaxFFError':
        """ Initializes the error function from classic ReaxFF files.

        Parameters
        ----------
        path
            Path to classic ReaxFF files, passed to :func:`.setup_reax_from_classic`.
        Inherited, validation_dataset scale_residuals
            See :class:`.BaseParamsError`.
        **kwargs
            Passed to :func:`.setup_reax_from_classic`.
        """
        dat_set, job_col, rxf_eng = setup_reax_from_classic(path, **kwargs)
        return cls(dat_set, job_col, rxf_eng, validation_dataset, scale_residuals)

    @classmethod
    def from_params_files(cls,
                          path: Union[Path, str],
                          validation_dataset: Optional[DataSet] = None,
                          scale_residuals: bool = False, ) -> 'ReaxFFError':
        """ Initializes the error function from ParAMS data files.

        Parameters
        ----------
        path
            Path to directory containing ParAMS data set, job collection and ReaxFF engine files (see
            :func:`.setup_reax_from_params`).
        Inherited, validation_dataset scale_residuals
            See :class:`.BaseParamsError`.
        """
        dat_set, job_col, rxf_eng = setup_reax_from_params(path)
        return cls(dat_set, job_col, rxf_eng, validation_dataset, scale_residuals)

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
            if PARAMS_VERSION_INFO == (0, 5, 0):
                allowed = np.array(self.par_eng._get_active())
            else:
                allowed = np.array([p.expose and p.name not in p.blacklist for p in self.par_eng])

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

    def get_grouping_matrix(self, *,
                            active_only: bool = False,
                            by_atoms: bool = False,
                            by_block: bool = False,
                            by_block_index: bool = False,
                            by_equation: bool = False,
                            by_description: bool = False) -> Tuple[List[Tuple], np.ndarray]:
        """ Returns a grouping matrix for use in the analysis tools (see :class:`.EstimatedEffects` and
        :attr:`.EstimatedEffects.groupings`). Various default groupings are provided.

        Parameters
        ----------
        active_only
            If :obj:`True`, the grouping will only be done for parameters which are 'active' otherwise it will be done
            for all parameters.

        by_atoms
            If :obj:`True`, parameter atom groups are included in the grouping.

            .. note::

               This groups by atom *group*, **not** atom *types*. This is because each parameter may only appear in
               one group, and an atom type will appear in several groups. Hence, this will result in a grouping like:

               >>> ['C', 'C.H', 'C.O', 'O.H', 'C.O.H', ...]

               not:

               >>> ['C', 'H', 'O']

        by_block
            If :obj:`True`, parameter block types are included in the grouping.

        by_block_index
            If :obj:`True`, the parameters' positions in the blocks are included in the grouping.

            .. warning::

               Do not use alone! Grouping by index without also grouping by block type will result in a senseless
               grouping.

        by_equation
            If :obj:`True`, the equations in which the parameters' appear are included in the grouping.

        by_description
            If :obj:`True`, the parameter descriptions are included in the grouping.

        Returns
        -------
        List
            List of unique group identifier 'names'.
        numpy.ndarray
            :math:`n \\times g` grouping matrix.
        """
        if PARAMS_VERSION_INFO < (0, 5, 1):
            raise ValueError("Method incompatible with ParAMS < v0.5.1")

        maps = []

        params = self.par_eng.active if active_only else self.par_eng

        if by_atoms:
            maps.append('atoms')
        if by_block:
            maps.append('block')
        if by_block_index:
            maps.append('block_index')
        if by_equation:
            maps.append('equation')
        if by_description:
            maps.append('description')

        signatures = [tuple(getattr(p, name) if name != 'atoms' else p.name.split(':')[0] for name in maps)
                      for p in params]

        bins = list(set(signatures))  # Unique ordered signatures

        group_matrix = np.array([[s == b for b in bins] for s in signatures], dtype=int)

        return bins, group_matrix


class XTBError(BaseParamsError):
    """ GFN-xTB error function. """

    @classmethod
    def from_params_files(cls,
                          path: Union[Path, str],
                          validation_dataset: Optional[DataSet] = None,
                          scale_residuals: bool = False, ) -> 'XTBError':
        """ Initializes the error function from ParAMS data files.

        Parameters
        ----------
        path
            Path to directory containing ParAMS data set, job collection and parameter engine files (see
            :func:`.setup_xtb_from_params`).
        Inherited, validation_dataset scale_residuals
            See :class:`.BaseParamsError`.
        """
        dat_set, job_col, rxf_eng = setup_xtb_from_params(path)
        return cls(dat_set, job_col, rxf_eng, validation_dataset, scale_residuals)

    def checkpoint_save(self, path: Union[Path, str]):
        """ Used to store files into a GloMPO checkpoint (at path) suitable to reconstruct the task when the checkpoint
        is loaded.
        """
        path = Path(path).resolve(True)
        self.dat_set.pickle_dump(str(path / 'data_set.pkl'))
        self.job_col.pickle_dump(str(path / 'job_collection.pkl'))
        self.par_eng.write(str(path))
