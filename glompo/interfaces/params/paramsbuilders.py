import warnings
from pathlib import Path
from typing import Tuple, Union

from scm.params.common._version import __version__ as PARAMS_VERSION
from scm.params.common.reaxff_converter import geo_to_params, trainset_to_params
from scm.params.core.dataset import DataSet
from scm.params.core.jobcollection import JobCollection
from scm.params.parameterinterfaces.reaxff import ReaxParams
from scm.plams.interfaces.adfsuite.reaxff import reaxff_control_to_settings

PARAMS_VERSION_INFO = tuple(map(int, PARAMS_VERSION.split('.')))


def setup_reax_from_classic(path: Union[Path, str], **kwargs) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """ Parses classic ReaxFF force field and configuration files into instances which can be evaluated by AMS.

    Parameters
    ----------
    path
        Path to directory containing classic ReaxFF configuration files:
    **kwargs
        Extra arguments passed to :func:`~scm.params.common.reaxff_converter.trainset_to_params`.

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
    dat_set = trainset_to_params(str(path / 'trainset.in'), **kwargs)

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


def setup_from_params(path: Union[Path, str], *,
                      jc_path: Union[Path, str, None] = None,
                      ts_path: Union[Path, str, None] = None,
                      pi_path: Union[Path, str, None] = None) -> Tuple[DataSet, JobCollection, ReaxParams]:
    """ Loads ParAMS produced ReaxFF files into ParAMS objects.

    Parameters
    ----------
    path
        A path to a directory containing:
            ``job_collection.yaml``
                Contains descriptions of the AMS jobs to evaluate. Must be of the form produced by
                :meth:`~scm.params.core.jobcollection.JobCollection.store`.
            ``training_set.yaml``
                Contains the description of the items in the training set. Must be of the form produced by
                :meth:`~scm.params.core.dataset.DataSet.store`.
            ``parameter_interface.yaml``
                Contains the parameter information produced by
                :meth:`~scm.params.parameterinterfaces.base.BaseParameters.yaml_store` representing
                the parameter values, metadata, and their ranges.
    jc_path
    ts_path
    pi_path
        Custom paths to the job collection, training set and parameter interface YAML files respectively if they do not
        have the expected default names or are not all in the same directory. If provided they will take precedence over
        `path` which will be the fallback for any ones which are not provided.
    """
    job_collection = JobCollection()
    training_set = DataSet()

    path = Path(path)
    jc_path = Path(jc_path) if jc_path else path / 'job_collection.yaml'
    ts_path = Path(ts_path) if ts_path else path / 'training_set.yaml'
    pi_path = Path(pi_path) if pi_path else path / 'parameter_interface.yaml'

    job_collection.load_yaml(str(jc_path))
    training_set.load(str(ts_path))
    reaxff_engine = ReaxParams.yaml_load(str(pi_path))

    return training_set, job_collection, reaxff_engine
