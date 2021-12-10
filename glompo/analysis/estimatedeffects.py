import copy
import inspect
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import dask
import dask.array as da
import numpy as np
import psutil

from ..common.wrappers import needs_optional_package

# Plot defaults
WIDTH = 14
HEIGHT = 7
FONTSIZE = 12
COLORS = ['#49DABF', '#FFBF69', '#ECC74C', '#404DEE']

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import colors
    from matplotlib import lines
    from matplotlib import patches
    from matplotlib import cm

    plt.rcParams['font.size'] = FONTSIZE
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['savefig.format'] = 'svg'
    plt.rcParams['figure.figsize'] = WIDTH, HEIGHT
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)

except (ModuleNotFoundError, ImportError, TypeError):  # TypeError caught for building docs
    pass

# Dask settings
dask.config.set(**{'array.slicing.split_large_chunks': False})

__all__ = ('EstimatedEffects',)

SpecialSlice = Union[None, int, str, List, slice, np.ndarray]


def pass_or_compute(func) -> Callable[..., Union[da.Array, np.ndarray]]:
    """ Wraps most methods in EstimatedEffects.
    Allows dask.arrays to be used internally amongst class methods but always return a numpy array to end users.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Union[da.Array, np.ndarray]:
        filename = inspect.stack()[1][1]
        call = func(*args, **kwargs)

        if filename == __file__:
            return call
        return call.compute()

    return wrapper


# noinspection PyIncorrectDocstring
class EstimatedEffects:
    """ Implementation of Morris screening strategy.
    Based on the original work of `Morris (1991) <https://doi.org/10.1080/00401706.1991.10484804>`_ but includes
    extensions published over the years. Global sensitivity method for expensive functions. Uses minimal number of
    function evaluations to develop a good proxy for the total sensitivity of each input factor. Produces three
    sensitivity measures (:math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`) that are able to capture magnitude and
    direction the sensitivity, as well as nonlinear and interaction effects. The user is directed to the references
    below for a detailed explanation of the meaning of each of these measures.

    Parameters
    ----------
    input_dims
        Number of factors in the input space on which the sensitivity analysis is being done.

    output_dims
        Number of dimensions in the response of the function.

    groupings
        :math:`k \\times g` array grouping each :math:`k` factor into one and only one of the :math:`g` groups.
        See :attr:`groupings`.

        .. warning::

           The use of groups comes at the cost of the :math:`\\mu` and :math:`\\sigma` metrics. They are unobtainable
           in this regime because it is not possible to define . Only :math:`\\mu^*` is accessible. Similarly, calls to
           functions relying on anything other than :math:`\\mu^*` will produce a warning and not be run.

    include_short_range
        If :obj:`True`, flags that the trajectories introduced to the calculation will include specially generated
        short-range points (see :mod:`.trajectories` for more details).

    convergence_threshold
        See :attr:`convergence_threshold`.

    cutoff_threshold
        See :attr:`ct`.

    References
    ----------
    Cosenza, A., Mannina, G., Vanrolleghem, P. A., & Neumann, M. B. (2013). Global sensitivity analysis in wastewater
    applications: A comprehensive comparison of different methods. Environmental Modelling & Software, 49, 40–52.
    https://doi.org/10.1016/J.ENVSOFT.2013.07.009

    Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large
    models. *Environmental Modelling & Software*, 22(10), 1509–1518. https://doi.org/10.1016/j.envsoft.2006.10.004

    Garcia Sanchez, D., Lacarrière, B., Musy, M., & Bourges, B. (2014). Application of sensitivity analysis in building
    energy simulations: Combining first- and second-order elementary effects methods. Energy and Buildings, 68(PART C),
    741–750. https://doi.org/10.1016/J.ENBUILD.2012.08.048

    Morris, M. D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments. *Technometrics*, 33(2),
    161–174. https://doi.org/10.1080/00401706.1991.10484804

    Ruano, M. V., Ribes, J., Seco, A., & Ferrer, J. (2012). An improved sampling strategy based on trajectory design for
    application of the Morris method to systems with many input factors. *Environmental Modelling & Software*, 37,
    103–109. https://doi.org/10.1016/j.envsoft.2012.03.008

    Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M., & Tarantola, S. (2007).
    Global Sensitivity Analysis. The Primer (A Saltelli, M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli,
    M. Saisana, & S. Tarantola (eds.)). *John Wiley & Sons, Ltd.* https://doi.org/10.1002/9780470725184

    Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban water
    quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*, 522,
    339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056

    Attributes
    ----------
    convergence_threshold : float
        Value of the Position Factor, below which sensitivities should be considered converged. See :meth:`is_converged`
        and :meth:`position_factor`.

    ct : float
        Value of :math:`\\mu^*` below which the factor is classified as 'non-influential'. See :attr:`classification`.

    dims : int
        The number of input factors for which sensitivity is being tested. Often referred to as :math:`k` throughout the
        documentation here to match literature. See :attr:`k`.

    has_short_range : bool
        :obj:`True` if the trajectories added to the calculation include specially generated short-range perturbations.

    out_dims : int
        Dimensionality of the output if one would like to investigate factor sensitivities against multiple function
        responses. Often referred to as :math:`h` in the documentation of equations. See :attr:`h`.

    outputs : numpy.ndarray
        :math:`r \\times (g+1) \\times h` array of function evaluations corresponding to the input factors in
        :attr:`trajectories`. Represents the responses of the model to the points in the :attr:`trajectories`. See
        :attr:`r`, :attr:`g` and :attr:`h`.

    trajectories : numpy.ndarray
        :math:`r \\times g+1 \\times k` array of :math:`r` trajectories, each with :math:`g+1` points of :math:`k`
        factors each. Represents the carefully sampled points in the factor space where the model will be evaluated.
        See :attr:`r`, :attr:`g` and :attr:`h`.
    """

    @property
    def r(self) -> int:
        """ The number of trajectories in the set. """
        return len(self.trajectories)

    @property
    def k(self) -> int:
        """ Pseudonym for :attr:`dims` """
        return self.dims

    @property
    def h(self) -> int:
        """ Pseudonym for :attr:`out_dims` """
        return self.out_dims

    @property
    def g(self) -> int:
        """ The number of factor groups. Equals :attr:`k` if analyzing each factor individually. """
        return self.groupings.shape[1]

    @property
    def groupings(self) -> np.ndarray:
        """ :math:`k \\times g` array grouping each :math:`k` factor into one and only one of the :math:`g` groups. This
        allows one to perform a sensitivity analysis with far fewer function evaluations and sensitivities are reported
        for groups rather than factors.

        .. note::

           This attribute cannot be altered. It is not possible to go from grouped analysis to a individual factor
           analysis (or vice versa), or to a different grouping due to the manner in which trajectories are constructed.
           One would need to start a new :class:`EstimatedEffects` instance and generate new trajectories to analyze a
           different grouping.

        Returns
        -------
        numpy.ndarray
            :math:`k \\times g` array mapping each factor to exactly one group.
        """
        return self._groupings

    @property
    def is_grouped(self) -> bool:
        """ :obj:`True` if a grouped factor analysis is being conducted. """
        return self._is_grouped

    @property
    @pass_or_compute
    def is_converged(self) -> da.Array:
        """ Converged if the instance has enough trajectories for the factor ordering to be stable.
        Returns :obj:`True` if the change in :meth:`position_factor` over the last 10 trajectory entries is smaller
        than :attr:`convergence_threshold`.

        Returns
        -------
        numpy.ndarray
            Array of length :math:`h` with boolean values indicating if the sensitivity metrics for that output have
            converged.
        """
        if self.r <= 10:
            return da.full(self.h, False, dtype=bool)
        return da.squeeze(da.absolute(self.position_factor(self.r - 10, self.r, 'all')) <= self.convergence_threshold)

    @property
    @pass_or_compute
    def mu(self) -> da.Array:
        """ Shortcut access to the Estimated Effects :math:`\\mu` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to: :code:`ee['mu', :, 'mean', :, 'all']`

        Returns
        -------
        numpy.ndarray
            :math:`h \\times k` array.

        Notes
        -----
        If using a single output, then this attribute is the metric itself. Using :code:`'mean'` has no effect.

        Warnings
        --------
        Unavailable if using groups, will raise a :obj:`ValueError`. See :attr:`groupings`.
        """
        return self['mu', :, 'mean', :].squeeze()

    @property
    @pass_or_compute
    def mu_star(self) -> da.Array:
        """ Shortcut access to the Estimated Effects :math:`\\mu^*` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to:
        :code:`ee['mu_star', :, 'mean', :, 'all']`

        Returns
        -------
        numpy.ndarray
            :math:`h \\times g` array.

        Notes
        -----
        If using a single output, then this attribute is the metric itself. Using :code:`'mean'` has no effect.
        """
        return self['mu_star', :, 'mean', :].squeeze()

    @property
    @pass_or_compute
    def sigma(self) -> da.Array:
        """ Shortcut access to the Estimated Effects :meth:`\\sigma` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to: :code:`ee['sigma', :, 'mean', :, 'all']`

        Returns
        -------
        numpy.ndarray
            :math:`h \\times k` array.

        Notes
        -----
        If using a single output, then this attribute is the metric itself. Using :code:`'mean'` has no effect.

        Warnings
        --------
        Unavailable if using groups, will raise a :obj:`ValueError`. See :attr:`groupings`.
        """
        return self['sigma', :, 'mean', :].squeeze()

    def __init__(self, input_dims: int,
                 output_dims: int,
                 groupings: Optional[np.ndarray] = None,
                 include_short_range: bool = False,
                 convergence_threshold: float = 0,
                 cutoff_threshold: float = 0.1):
        self.dims: int = input_dims
        self.out_dims = output_dims
        self.has_short_range = include_short_range
        self._groupings = groupings if groupings is not None else np.identity(input_dims)
        self._is_grouped = groupings is not None
        if np.sum(self._groupings) != input_dims:
            raise ValueError("Invalid grouping matrix, each factor must be in exactly 1 group.")

        n = 2 if include_short_range else 1
        with warnings.catch_warnings():  # Catch divide by zero warning when creating empty dask array
            warnings.simplefilter('ignore', RuntimeWarning, 2894)
            self.trajectories = da.empty((0, n * self.g + 1, self.k))
            self.outputs = da.empty((0, n * self.g + 1, self.h))

        self.convergence_threshold = convergence_threshold
        self.ct = cutoff_threshold
        self._metrics = {'short': np.array([[[]]]),
                         'long': np.array([[[]]]),
                         'all': np.array([[[]]])}

        # Detect operating context
        if '__IPYTHON__' in globals():
            self._is_ipython = True
        else:
            self._is_ipython = False

    @pass_or_compute
    def __getitem__(self, item) -> da.Array:
        """ Retrieves the sensitivity metrics (:math:`\\mu, \\mu^*, \\sigma`) for a particular calculation configuration
        The indexing is a strictly ordered set of maximum length 5:

        * First index (:code:`metric_index`):
            Indexes the metric. Also accepts :obj:`str` which are paired to the following corresponding :obj:`int`:

            === ===
            int str
            === ===
            0   'mu'
            1   'mu_star'
            2   'sigma'
            === ===

        * Second index (:code:`factor_index`):
            Indexes the factor or group for which sensitivities are being calculated.

        * Third index (:code:`out_index`):
            Only applicable if a multidimensional output is being investigated. Determines which metrics to use in the
            calculation. Accepts one or a combination of the following:

               * Any integers or slices of 0, 1, ... :math:`h`: The metrics for the outputs at the
                 corresponding index will be used.

               * :code:`'mean'`: The metrics will be averaged across the output dimension.

        * Fourth Index (:code:`traj_index`):
            Which trajectories to use in the calculation.

        * Fifth Index (:code:`range_key`):
            Indexes the types of points to use. Accepts only one :obj:`str` value. Defaults to :code:`'all'`, will raise
            an error if anything other than :code:`'all'` is used and :attr:`has_short_range` is :obj:`False`.

            ======= ===
            str     Description
            ======= ===
            'short' Only analyze 'short-range' distances specially added to the trajectories by the make trajectory
                    method if its parameter :code:`include_short_range=True`.
            'long'  Only analyze the standard, longer-range, trajectory points.
            'all'   Include all points in the calculation.
            ======= ===

        Parameters
        ----------
        item
            Tuple of slices and indices. Maximum length of 5.

        Returns
        -------
        numpy.ndarray
            Three dimensional array of selected metrics, factors/groups and outputs.

        Raises
        ------
        ValueError
            If an attempt is made to access :math:`\\mu` or :math:`\\sigma` while using groups.

        Notes
        -----
        For all of the indices using :code:`:`, :code:`'all'` or :obj:`None` will return all items.

        If an index is not supplied, all indices will also be returned.

        When passing indices as arguments in other functions (see :meth:`ranking` for example) it is not possible to
        pass slices exactly as is done here (e.g. :code:`:2` would not be a valid construct as a function argument). To
        pass a slice, use a :obj:`slice` object (e.g. :code:`:2` could be sent as :code:`slice(None, 2)`).

        If `traj_index` is provided then the results will *not* be saved as the :attr:`mu`, :attr:`mu_star` and
        :attr:`sigma` attributes of the class. If all available trajectories are used *and* metrics for all the outputs
        are calculated, then the results are saved into the above mentioned attributes.

        If a user attempts to access any of the attributes (and the number of trajectories has changed since the last
        call), this method is automatically called for all available trajectories. In other words, there is never any
        risk of accessing metrics which are out-of-sync with the number of trajectories appended to the calculation.

        Examples
        --------
        Return all metrics for all factors and outputs, using all available trajectories:

        >>> ee[]

        Return :math:`\\mu^*` for the second factor, averaged over all outputs:

        >>> ee[1, 1, 'mean']

        Return :math:`\\mu` and :math:`\\sigma` metrics for the first three factors and output 6:

        >>> ee[['mu', 'sigma'], 0:3, 6]

        Return the metrics using the first 20 trajectories only:

        >>> ee[:, :, :, :20]

        Return :math:`\\mu^*`, averaged over all factors, using only analysis of the short-range points:

        >>> ee[1, :, 'mean', :, 'short']

        """
        if self.trajectories.size == 0:
            warnings.warn("Please add at least one trajectory before attempting to access calculations.", UserWarning)
            return da.array([[[]]])

        if not isinstance(item, tuple):
            item = (item,)  # NOT the same as tuple(item) since you don't want to expand 'mu' to ('m', 'u')
        item += (None,) * (5 - len(item))
        m = self._expand_index('metric', item[0])
        k = self._expand_index('factor', item[1])
        h = self._expand_index('output', item[2])
        t = self._expand_index('trajec', item[3])

        assert 'all' not in h

        range_key = item[4] if item[4] is not None else 'all'
        if not any((range_key == allowed for allowed in ('short', 'long', 'all'))):
            raise ValueError(f"Cannot parse '{item[4]}', only 'all', 'short', 'long' allowed.")

        if self.is_grouped and any([i in m for i in [0, 2]]):
            raise ValueError('Cannot access mu and sigma metrics if groups are being used')

        all_t = list(range(self.r))
        all_h = list(range(self.h))

        orig_h = copy.copy(h)
        spec_out = 'mean' in h
        h = all_h if spec_out else h

        # Attempt to access existing results
        if t == all_t and self._metrics[range_key].size > 0:
            metrics = da.array(self._metrics[range_key][np.ix_(m, k, h)])
        else:
            metrics = self._calculate_metrics(h, t, range_key)
            if h == all_h and t == all_t and metrics.nbytes < 0.3 * psutil.virtual_memory().available:
                # Save results to cache if a full calculation was done and can fit in memory
                self._metrics[range_key] = metrics.compute()
            metrics = metrics[m][:, k]  # Dask 2021.3 doesn't support better slicing, >2021.3 not supported in Python3.6

        if spec_out:
            cols = []
            for o in orig_h:
                if o == 'mean':
                    cols.append(da.mean(metrics, 2)[:, :, None])
                else:
                    cols.append(metrics[:, :, o, None])
            metrics = da.concatenate(cols, axis=2)

        return metrics

    def add_trajectory(self, trajectory: np.ndarray, outputs: np.ndarray):
        """ Add a trajectory of points and their corresponding model output to the calculation.

        Parameters
        ----------
        trajectory
            One or several trajectories of points as produced by one of the trajectory generation functions
            (see :mod:`.trajectories`). Should have a shape of :math:`n \\times (pg+1) \\times k` where :math:`k`
            is the number of factors / dimensions of the input and :math:`g` is the number of groups being analyzed, and
            :math:`n` is the number of trajectories being appended. :math:`p` is 1, or 2 if short range analysis is also
            being conducted.
        outputs
            :math:`n \\times (pg+1) \\times h` model outputs corresponding to the points in `trajectory`. Where
            :math:`h` is the dimensionality of the outputs.

        Raises
        ------
        ValueError
            If `trajectory` or `outputs` do not match the dimensions above.

        See Also
        --------
        :attr:`g`
        :attr:`groupings`
        :attr:`k`

        Notes
        -----
        The actual calculation of the Estimated Effects metrics is not performed in this method. Adding new trajectories
        is essentially free. The calculation is only performed the moment the user attempts to access any of the
        metrics. The results of the calculation are held in memory, thus if the number of trajectories remains
        unchanged, the user may continue accessing the metrics at no further cost.
        """
        p = int(self.has_short_range) + 1

        if trajectory.ndim != 3:
            trajectory = np.array([trajectory])

        if outputs.ndim != 3:
            outputs = np.array([outputs]).reshape((-1, p * self.g + 1, self.h))

        if trajectory.shape[1:] != (p * self.g + 1, self.k):
            raise ValueError(f"Cannot parse trajectory with shape {trajectory.shape}, must be (n, {p * self.g + 1}, "
                             f"{self.k}).")
        if self.h > 1 and outputs.shape[1:] != (p * self.g + 1, self.h):
            raise ValueError(f"Cannot parse outputs with shape {outputs.shape}, must be (n, {p * self.g + 1}, "
                             f"{self.h})")

        if self.h == 1 and outputs.shape[1:] != (p * self.g + 1, self.h) and outputs.shape[1:] != (p * self.g + 1,):
            raise ValueError(f"Cannot parse outputs with shape {outputs.shape}, (n, {p * self.g + 1}) expected.")

        if self.h == 1:
            outputs = outputs.reshape((-1, p * self.g + 1, self.h))

        # Clear old results
        self._metrics = {'short': np.array([[[]]]),
                         'long': np.array([[[]]]),
                         'all': np.array([[[]]])}

        self.trajectories = da.append(self.trajectories, trajectory, axis=0)
        self.outputs = da.append(self.outputs, outputs, axis=0)

    @pass_or_compute
    def elementary_effects(self,
                           out_index: SpecialSlice = None,
                           traj_index: Union[int, slice, List[int]] = None,
                           range_key: str = 'all') -> da.Array:
        """ Returns the raw elementary effects for a given calculation configuration.

        Parameters
        ----------
        Inherited, out_index traj_index range_key
            See :meth:`__getitem__`.

        Returns
        -------
        numpy.ndarray
            :math:`r \\times g \\times h` of raw elementary effects from which the metrics are calculated.
        """
        h = self._expand_index('output', out_index)
        t = self._expand_index('trajec', traj_index)

        ee = self._calculate_ee(h, t, range_key)

        return ee

    @pass_or_compute
    def position_factor(self, i: int, j: int, out_index: SpecialSlice = 'mean', range_key: str = 'all') -> da.Array:
        """ Returns the position factor metric.
        This is a measure of convergence. Measures the changes between the factor rankings obtained when using `i`
        trajectories and `j` trajectories.  Where `i` and `j` are a number of trajectories such that
        :math:`0 < i < j \\leq M` where :math:`M` is the number of trajectories added to the calculation.

        The position factor metric (:math:`PF_{r_i \\to r_j}`) is calculated as:

        .. math::

           PF_{r_i \\to r_j} = \\sum_{k=1}^k \\frac{2\\left|P_{k,i} - P_{k,j}\\right|}{P_{k,i} + P_{k,j}}

        where:

           :math:`P_{k,i}` is the ranking of factor :math:`k` using :math:`i` trajectories.

           :math:`P_{k,j}` is the ranking of factor :math:`k` using :math:`j` trajectories.

        Parameters
        ----------
        i
            Initial trajectory index from which to start the comparison.
        j
            Final trajectory index against which the comparison is made.
        Inherited, out_index range_key
            See :meth:`__getitem__`.

        Returns
        -------
        numpy.ndarray
            Array equal in length to the number of outputs requested.

        References
        ----------
        Ruano, M. V., Ribes, J., Seco, A., & Ferrer, J. (2012). An improved sampling strategy based on trajectory design
        for application of the Morris method to systems with many input factors. *Environmental Modelling & Software*,
        37, 103–109. https://doi.org/10.1016/j.envsoft.2012.03.008

        Cosenza, A., Mannina, G., Vanrolleghem, P. A., & Neumann, M. B. (2013). Global sensitivity analysis in
        wastewater applications: A comprehensive comparison of different methods. Environmental Modelling & Software,
        49, 40–52. https://doi.org/10.1016/J.ENVSOFT.2013.07.009
        """
        pos_i = da.atleast_2d(self.ranking(out_index, slice(None, i), range_key))
        pos_j = da.atleast_2d(self.ranking(out_index, slice(None, j), range_key))
        return da.sum(2 * da.absolute(pos_i - pos_j) / (pos_i + pos_j), axis=1).squeeze()

    def order_factors(self,
                      out_index: SpecialSlice = 'mean',
                      traj_index: SpecialSlice = None,
                      range_key: str = 'all',
                      n_bootstrap_samples: int = 1) -> np.ndarray:
        """ Returns factor indices in descending order of their influence on the outputs.
        The *positions* in the array are the rankings, the *contents* of the array are the factor / group indices. This
        is the inverse of :meth:`ranking`.

        Parameters
        ----------
        Inherited, out_index traj_index range_key
            See :meth:`__getitem__`.
        n_bootstrap_samples
            Optional. If greater than 1 the rankings will be generated multiple times using a bootstrap resampling
            method, otherwise the raw data will be used. Supersedes `traj_index` if also supplied.

        Returns
        -------
        numpy.ndarray
            Array of :math:`g` length vectors for each of the selected outputs.

        See Also
        --------
        :meth:`ranking`

        Examples
        --------
        Return a factor ordering for all outputs, using all available trajectories:

        >>> ee.order_factors()

        Returns the factor ordering of the third output using the first 10 trajectories:

        >>> ee.order_factors(2, slice(None, 10))
        """
        if n_bootstrap_samples > 1:
            metric = da.array([self[1, :, out_index, np.sort(np.random.choice(self.r, self.r, replace=True)), range_key]
                               for _ in range(n_bootstrap_samples)]).mean(0)
        else:
            metric = da.atleast_3d(self[1, :, out_index, traj_index, range_key])
        return metric.compute().argsort(1)[:, -1::-1].squeeze().T  # Dask doesn't support search

    def ranking(self,
                out_index: SpecialSlice = 'mean',
                traj_index: SpecialSlice = None,
                range_key: str = 'all',
                n_bootstrap_samples: int = 1) -> np.ndarray:
        """ Returns the ranking of each factor being analyzed.
        The *positions* in the array are the factor or group indices, the *contents* of the array are rankings such that
        1 is the most influential factor / group and :math:`g+1` is the least influential. This is the inverse of
        :meth:`order_factors`.

        Parameters
        ----------
        Inherited, out_index traj_index range_key
            See :meth:`__getitem__`.
        n_bootstrap_samples
            See :meth:`order_factors`

        Returns
        -------
        numpy.ndarray
            Array of :math:`g` length vectors for each of the selected outputs.

        See Also
        --------
        :meth:`order_factors`
        :meth:`plot_bootstrap_rankings`
        """
        order_factors = self.order_factors(out_index, traj_index, range_key, n_bootstrap_samples)
        return np.squeeze(np.atleast_2d(order_factors).argsort(1) + 1)  # Dask doesn't support search

    def classification(self, n_cats: int, out_index: Union[int, str] = 'mean',
                       range_key: str = 'all') -> Dict[str, np.ndarray]:
        """ Returns a dictionary with each factor index classified according to its effect on the function outputs.

        Parameters
        ----------
        n_cats
            Number of categories in the classification. Two categorizations from literature are supported:

            * :code:`n_cats = 3`:

              Three category classification of `Vanrolleghem et al (2015)
              <https://doi.org/10.1016/J.JHYDROL.2014.12.056>`_:

              ========================= ============== ======================= ============================
              Name                      Condition                              Description
              ------------------------- -------------------------------------- ----------------------------
              ..                        :math:`\\mu^*`  :math:`\\sigma/\\mu^*`
              ========================= ============== ======================= ============================
              :code:`'non-influential'` < CT                                   No appreciable effect.
              :code:`'important'`       > CT           < :math:`\\sqrt{r}/2`    Strong individual effect.
              :code:`'interacting'`     > CT           > :math:`\\sqrt{r}/2`    Strong interacting effects.
              ========================= ============== ======================= ============================

            * :code:`n_cats = 5`:

              Five category classification of `Garcia Sanchez et al. (2014)
              <https://doi.org/10.1016/J.ENBUILD.2012.08.048>`_:

              ========================= ============== ======================= ============================
              Name                      Condition                              Description
              ------------------------- -------------------------------------- ----------------------------
              ..                        :math:`\\mu^*`  :math:`\\sigma/\\mu^*`
              ========================= ============== ======================= ============================
              :code:`'non-influential'` < CT                                   No appreciable effect.
              :code:`'linear'`          > CT           < 0.1                   Strong linear effect.
              :code:`'monotonic'`       > CT           [0.1, 0.5)              Strong monotonic effect.
              :code:`'quasi-monotonic'` > CT           [0.5, 1.0)              Moderate monotonic effect.
              :code:`'interacting'`     > CT           > 1.0                   Strongly non-linear effects.
              ========================= ============== ======================= ============================

        out_index
            Output dimension along which to do the classification.

        range_key
            See :meth:`__getitem__`.

        Returns
        -------
        Dict[str, numpy.ndarray]
            Dictionary with the category names as keys. The corresponding arrays are two-dimensional: number of outputs
            requested by `out_index` by number of parameters in the category.

        Raises
        ------
        ValueError
            If `out_index` includes more than one index.

        ValueError
            If `n_cats` does not equal 3 or 5.

        Warnings
        --------
        Unavailable if using groups, will raise a :obj:`ValueError`. See :attr:`groupings`.

        Notes
        -----
        `out_index` supports :code:`'mean'` (averages metrics over the outputs) and integer indices. Unlike most other
        methods, however, it does not support slices and combinations of outputs. Only one classification can be
        generated at a time. The same is true for `range_key` where only one choice is supported at a time.

        References
        ----------
        Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban
        water quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*,
        522, 339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056

        Garcia Sanchez, D., Lacarrière, B., Musy, M., & Bourges, B. (2014). Application of sensitivity analysis in
        building energy simulations: Combining first- and second-order elementary effects methods. Energy and Buildings,
        68(PART C), 741–750. https://doi.org/10.1016/J.ENBUILD.2012.08.048
        """
        if out_index != 'mean' and not isinstance(out_index, int):
            raise ValueError('Classification can only be done on a single output at a time.')
        mu, ms, sd = self[:, :, out_index, :, range_key].squeeze().compute()

        fr = np.zeros_like(mu)
        np.divide(sd, ms, out=fr, where=sd != 0)

        if n_cats == 3:
            grad = np.sqrt(self.r) / 2
            classi = {'important': np.argwhere((ms > self.ct) & (fr < grad)).ravel(),
                      'interacting': np.argwhere((ms > self.ct) & (fr >= grad)).ravel(),
                      'non-influential': np.argwhere(ms < self.ct).ravel()}

        elif n_cats == 5:
            classi = {'non-influential': np.argwhere(ms < self.ct).ravel(),
                      'linear': np.argwhere((ms > self.ct) & (fr < 0.1)).ravel(),
                      'monotonic': np.argwhere((ms > self.ct) & (fr >= 0.1) & (fr < 0.5)).ravel(),
                      'quasi-monotonic': np.argwhere((ms > self.ct) & (fr >= 0.5) & (fr < 1.0)).ravel(),
                      'interacting': np.argwhere((ms > self.ct) & (fr >= 1)).ravel(),
                      }

        else:
            raise ValueError(f'Cannot parse n_cats = {n_cats}, must equal 3 or 5.')

        assert sum([c.size for c in classi.values()]) == self.g
        return classi

    def relevance_factor(self, n_cats: int, category: str, out_index: Union[int, str] = 'mean',
                         range_key: str = 'all') -> np.ndarray:
        """ Measure of the fraction of factors which have an influence on the outputs.
        Calculated as:

        .. math::

           Rel_{\\text{category}} = \\frac{N_\\text{category}}{g}

        Where :math:`N_\\text{category}` are the number of factors classified into a particular classification category.

        Parameters
        ----------
        n_cats
            See :meth:`classification`.
        category
            Classification key (see :meth:`classification`) to calculate the factor for.
        out_index
            See :meth:`classification`.
        range_key
            See :meth:`classification`.

        References
        ----------
        Cosenza, A., Mannina, G., Vanrolleghem, P. A., & Neumann, M. B. (2013). Global sensitivity analysis in
        wastewater applications: A comprehensive comparison of different methods. Environmental Modelling & Software,
        49, 40–52. https://doi.org/10.1016/J.ENVSOFT.2013.07.009
        """
        return self.classification(n_cats, out_index, range_key)[category].size / self.g

    @pass_or_compute
    def bootstrap_metrics(self,
                          n_samples: int,
                          metric_index: SpecialSlice = None,
                          factor_index: SpecialSlice = None,
                          out_index: SpecialSlice = None,
                          range_key: str = 'all') -> da.Array:
        """ Calculates sensitivity metrics with a confidence interval based on resampling bootstrapping.

        Parameters
        ----------
        n_samples
            Number of resamples to perform.
        Inherited, metric_index factor_index out_index range_key
            See :meth:`__getitem__`

        Returns
        -------
        numpy.ndarray
            Two three-dimensional arrays of selected metrics, factors/groups and outputs.
            Metrics are ordered: :math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`.
            The first array contains the mean value of the bootstrap, the second contains its standard deviation.
        """
        multis = da.array([self[metric_index, factor_index, out_index,
                                np.sort(np.random.choice(self.r, self.r, replace=True)),
                                range_key] for _ in range(n_samples)])
        return da.array([multis.mean(0), multis.std(0)])

    @needs_optional_package('matplotlib')
    def plot_sensitivities(self,
                           out_index: SpecialSlice = 'mean',
                           range_key: Union[str, Sequence[str]] = 'all',
                           factor_labels: Union[None, Sequence[str], str] = None,
                           out_labels: Union[None, Sequence[str], str] = None,
                           log_scale: bool = False,
                           path: Union[None, Path, str] = None) -> Union[plt.Figure, List[plt.Figure]]:
        """ Produces a sensitivity plot.
        Produces two side-by-side scatter plots. The first is :math:`\\mu^*` versus :math:`\\sigma`, the second is
        :math:`\\mu^*` versus :math:`\\sigma/\\mu^*`. Defined dividers are included to classify factors into:

           #. 'important'
           #. 'linear'
           #. 'monotonic'
           #. 'interacting'
           #. 'non-influential'

        Parameters
        ----------
        out_index
            See :meth:`__getitem__`. If :obj:`None` or :code:`'all'`, one plot will be created for each output.
        range_key
            See :meth:`__getitem__`. Accepts a single key or a list of any combination of :code:`'all'`,
            :code:`'short'`, :code:`'long'`. If multiple keys are provided, they are plotted on image rows in the order
            provided.
        factor_labels
            Optional sequence of descriptive names for each factor to add to the figure. Defaults to the factor's
            index position.
        out_labels
            Optional list of names to give to the outputs, if multiple are selected. Will be used in the figure title
            and as the filename. Must be equal in length to `out_index`.
        log_scale
            If :obj:`True` the axes will be plotted on a log scale.
        path
            Optional path to which the plot will be saved. Default is :obj:`None` which does not save the plot to disk.

            If a path is provided and one plot is being produced (see `out_index`) this is interpreted as the filename
            with which to save the figure.

            If multiple plots are produced for all the outputs then this is interpreted as a directory into
            which the figures will be saved.

        Returns
        -------
        Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]
            Object or objects which contains the plot/s, allowing the user to further customize them to suit their
            needs.

            .. note::

               :class:`matplotlib.figure.Figure` objects are the highest level interface to the images, providing access
               to all the lower level components. Users will most often need to access the :class:`matplotlib.axes.Axes`
               objects to make the most common changes. Note, that most :class:`~matplotlib.figure.Figure` instances
               contain several :class:`~matplotlib.axes.Axes` instances. See example:

                >>> fig = ee.plot_sensitivities()
                >>> ax = fig.get_axes()
                >>> ax[0].set_title('My Custom Title')


        Warnings
        --------
        Unavailable if using groups, will raise a :obj:`ValueError`. See :attr:`groupings`.

        If using `log_scale` parameters where :math:`\\sigma = \\mu^* = 0` will not appear on the plot.

        References
        ----------
        Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban
        water quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*,
        522, 339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056

        Garcia Sanchez, D., Lacarrière, B., Musy, M., & Bourges, B. (2014). Application of sensitivity analysis in
        building energy simulations: Combining first- and second-order elementary effects methods. Energy and Buildings,
        68(PART C), 741–750. https://doi.org/10.1016/J.ENBUILD.2012.08.048
        """
        return self._plotting_core(out_index=out_index,
                                   plot_stub=self._plot_sensitivities_stub,
                                   range_key=range_key,
                                   factor_labels=factor_labels,
                                   out_labels=out_labels,
                                   log_scale=log_scale,
                                   path=path)

    @needs_optional_package('matplotlib')
    def plot_rankings(self,
                      out_index: SpecialSlice = 'mean',
                      range_key: Union[str, Sequence[str]] = 'all',
                      truncate_after: Optional[None] = None,
                      factor_labels: Union[None, Sequence[str], str] = None,
                      out_labels: Union[None, Sequence[str], str] = None,
                      log_scale: bool = False,
                      path: Union[None, Path, str] = None) -> Union[plt.Figure, List[plt.Figure]]:
        """ Produces the factor rankings as a plot.
        If a single `range_key` is used then the method plots the ordered :math:`\\mu^*` values against their
        corresponding parameter indices.

        If two `range_keys` are provided then :math:`\\mu^*` for each are plotted against one another. Points which do
        not lie on the diagonal identify outliers which behave differently over short and long distances.

        Parameters
        ----------
        Inherited, out_index factor_labels out_labels log_scale path
            See :meth:`plot_sensitivities`.
        range_key
            Accepts either one or two of the allowed range keys: :code:`'all'`, :code:`'short'` and :code:`'long'`.
        truncate_after
            The number of factors to include in the plot. Any factors which are ranked higher than this are excluded.
            Used to make analyses with many factors legible. Nothing is truncated by default.

            .. note::

               Only effects the single `range_key` plot.

        Returns
        -------
        Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]
            See :meth:`plot_sensitivities`.
        """
        if isinstance(range_key, str) or len(range_key) == 1:
            stub = self._plot_single_ranking_stub
        elif len(range_key) == 2:
            stub = self._plot_double_ranking_stub
        else:
            raise ValueError("Only 1 or 2 range keys can be plotted at the same time.")

        return self._plotting_core(out_index=out_index,
                                   plot_stub=stub,
                                   factor_labels=factor_labels,
                                   out_labels=out_labels,
                                   log_scale=log_scale,
                                   range_key=range_key,
                                   path=path,
                                   truncate_after=truncate_after)

    @needs_optional_package('matplotlib')
    def plot_convergence(self,
                         out_index: SpecialSlice = 'mean',
                         range_key: Union[str, Sequence[str]] = 'all',
                         step_size: int = 10,
                         out_labels: Union[None, Sequence[str], str] = None,
                         path: Union[None, Path, str] = None) -> plt.Figure:
        """ Plots the evolution of the Position Factor (:math:`PF_{r_i \\to r_j}`) metric as a function of increasing
        number of trajectories.

        Parameters
        ----------
        out_index
            See :meth:`__getitem__`. If multiple output dimensions are selected, they will be included on the same plot
        range_key
            See :meth:`__getitem__`.
        step_size
            The step size in number of trajectories when calculating the position factor.
        out_labels
            Optional sequence of descriptive labels for the plot legend corresponding to the outputs selected to be
            plotted. Defaults to 'Output 0', 'Output 1', 'Output 2', ...
        path
            Optional path to which the plot will be saved. Default is None, whose behavior is context dependent:

               * If in an interactive IPython or Jupyter context, plots will be shown and not saved.

               * Otherwise, the plot is saved with the default name.

        Returns
        -------
        Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]
            See :meth:`plot_sensitivities`

        Notes
        -----
        The Position Factor metric is a measure of how much rankings have changed between the rankings calculated
        using :math:`i` trajectories, and the rankings calculated using :math:`j` trajectories sometime later. Thus, if
        `step_size` is 10 then the plot would show the evolution of the Position Factor at:
        :math:`1 \\to 10, 10 \\to 20, 20 \\to 30, ...`

        See Also
        --------
        :meth:`position_factor`
        """
        fig, ax = plt.subplots()

        ax.set_title('Convergence of sensitivity rankings ($PF$ v $r$)')
        ax.set_xlabel('Trajectories Compared ($i \\to j$)')
        ax.set_ylabel('Position Factor ($PF_{i \\to j}$)')

        steps = np.clip([(i, i + step_size) for i in range(1, self.r, step_size)], 1, self.r)

        if isinstance(range_key, str):
            range_key = [range_key]

        labs = []
        for rk in range_key:
            pf = da.array([da.atleast_1d(self.position_factor(pair[0], pair[1], out_index, rk)) for pair in steps])
            pf = pf.compute()
            plot_lines = ax.plot(pf,
                                 marker={'all': 'o', 'long': 'x', 'short': 'd'}[rk],
                                 linestyle={'all': '-', 'long': '--', 'short': ':'}[rk])
            for i, l in enumerate(plot_lines):
                l.set_color(COLORS[i])

            if pf.shape[1] > 1 or len(range_key) > 1:
                if out_labels:
                    out_labels = [out_labels] if isinstance(out_labels, str) else out_labels
                    lab = [f'{ol} ({rk})' for ol in out_labels]
                    assert len(lab) == pf.shape[1], \
                        "Number of out labels does not match the number of out indices to be plotted."
                    labs += lab
                else:
                    labs += [f'Output {i} ({rk})' for i in range(pf.shape[1])]

        ax.legend(labels=labs)

        ax.set_xticks([i for i, _ in enumerate(steps)])
        ax.set_xticklabels([f'{s[0]}$\\to${s[1]}' for s in steps], rotation=45)

        fig.tight_layout()
        if path:
            fig.savefig(path, transparent=False, facecolor='white')

        plt.close()
        return fig

    @needs_optional_package('matplotlib')
    def plot_bootstrap_metrics(self,
                               n_samples: int = 10,
                               metric_index: SpecialSlice = None,
                               factor_index: SpecialSlice = None,
                               out_index: SpecialSlice = None,
                               range_key: str = 'all',
                               log_scale: bool = False,
                               out_labels: Union[None, Sequence[str], str] = None,
                               factor_labels: Union[None, Sequence[str], str] = None,
                               path: Union[None, Path, str] = None) -> Union[plt.Figure, List[plt.Figure]]:
        """ Plots the results of a boostrap analysis on the metrics.

        Parameters
        ----------
        n_samples
            See :meth:`bootstrap_metrics`.
        Inherited, metric_index factor_index out_index
            See :meth:`__getitem__`.
        range_key
            See :meth:`__getitem__`. Only a single key is allowed at a time.
        log_scale
            If :obj:`True` the metric axis will be plotted on a log scale.
        out_labels
            Optional list of names to give to the outputs, if multiple are selected. Will be used in the figure title
            and as the filename. Must be equal in length to `out_index`.
        factor_labels
            Optional list of names to gives the factors. Will be used in the axes labels. Must be equal in length to
            `factor_index`.
        path
            See :meth:`plot_sensitivities`.

        Returns
        -------
        Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]
            See :meth:`plot_sensitivities`.
        """
        assert any([range_key == rk for rk in ('all', 'short', 'long')]), \
            "Only a single choice of 'all', 'short' or 'long' is supported."

        if self.is_grouped and metric_index is None:
            metric_index = 'mu_star'
            met_map = {0: '$\\mu^*$'}
        else:
            metric_index = self._expand_index('metric', metric_index)
            met_map = {i: {0: '$\\mu$', 1: '$\\mu^*$', 2: '$\\sigma$'}[m] for i, m in enumerate(metric_index)}

        factor_index = self._expand_index('factor', factor_index)
        out_index = self._expand_index('output', out_index)

        if factor_labels:
            if isinstance(factor_labels, str):
                factor_labels = [factor_labels]
            assert len(factor_labels) == len(factor_index), \
                "Number of factor labels does not match the number of factor indices to be plotted."
        else:
            factor_labels = factor_index.copy()

        if out_labels:
            if isinstance(out_labels, str):
                out_labels = [out_labels]
            assert len(out_labels) == len(out_index), \
                "Number of out labels does not match the number of out indices to be plotted."
        else:
            out_labels = out_index.copy()

        boot = self.bootstrap_metrics(n_samples, metric_index, factor_index, out_index, range_key)
        boot_m, boot_s = boot.compute()
        metrics = self[metric_index, factor_index, out_index, :, range_key].compute()

        is_multi = False
        if path:
            path = Path(path)
            if boot_m.shape[2] > 1:
                path.mkdir(exist_ok=True, parents=True)
                is_multi = True

        figs = []
        for o, oname in enumerate(out_labels):
            fig, ax = plt.subplots(boot_m.shape[0], 1, figsize=(WIDTH, HEIGHT * boot_m.shape[0]))
            if boot_m.shape[0] == 1:
                ax = [ax]

            for m in range(boot_m.shape[0]):
                ax[m].errorbar(factor_index, boot_m[m, :, o],
                               yerr=boot_s[m, :, o], fmt='o', ecolor='k', label='Bootstrap')
                ax[m].scatter(factor_index, metrics[m, :, o],
                              color=COLORS[3], marker='_', zorder=1000, label='Raw Result')
                ax[m].set_xlabel("Parameter")
                ax[m].set_ylabel(met_map[m], fontsize=int(1.5 * FONTSIZE))
                ax[m].legend()
                ax[m].set_yscale('log' if log_scale else 'linear')

                ax[m].set_xticks(factor_index)
                ax[m].set_xticklabels(factor_labels, rotation=30)

            name = oname if not isinstance(oname, int) else f'{oname:03}'
            ax[0].set_title(name + f"\n(Number of resamples: {n_samples})" + f"\n(Using {range_key} points)")

            fig.tight_layout()
            if path:
                fig.savefig(path / name if is_multi else path, transparent=False, facecolor='white')

            plt.close()
            figs.append(fig)

        if len(figs) == 1:
            return figs[0]
        return figs

    def plot_bootstrap_rankings(self,
                                n_samples: int = 10,
                                out_index: SpecialSlice = None,
                                range_key: str = 'all',
                                truncate_after: Optional[int] = None,
                                out_labels: Union[None, Sequence[str], str] = None,
                                factor_labels: Union[None, Sequence[str], str] = None,
                                path: Union[None, Path, str] = None) -> Union[plt.Figure, List[plt.Figure]]:
        """ Plots the results of a boostrap analysis on the factor rankings.

        Parameters
        ----------
        n_samples
            See :meth:`bootstrap_metrics`.
        out_index
            See :meth:`__getitem__`.
        range_key
            See :meth:`__getitem__`. Only a single key is allowed at a time.
        truncate_after
            The number of factors to include in the plot. Any factors which are ranked higher than this are excluded.
            Used to make analyses with many factors legible. Nothing is truncated by default.
        out_labels
            Optional list of names to give to the outputs, if multiple are selected. Will be used in the figure title
            and as the filename. Must be equal in length to `out_index`.
        factor_labels
            Optional list of names of length :attr:`g` to gives the factors. Will be used in the axes labels.
        path
            See :meth:`plot_sensitivities`.

        Returns
        -------
        Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]
            See :meth:`plot_sensitivities`.
        """
        nodes = [0, 1 / n_samples, 1]
        cmap = colors.LinearSegmentedColormap.from_list("glompo",
                                                        [*zip(nodes, ['#FFFFFF', COLORS[1], COLORS[3]])])

        assert any([range_key == rk for rk in ('all', 'short', 'long')]), \
            "Only a single choice of 'all', 'short' or 'long' is supported."

        out_index = self._expand_index('output', out_index)

        if factor_labels:
            if isinstance(factor_labels, str):
                factor_labels = [factor_labels]
            assert len(factor_labels) == self.g, \
                "Number of factor labels does not match the number of factors."

        if out_labels:
            if isinstance(out_labels, str):
                out_labels = [out_labels]
            assert len(out_labels) == len(out_index), \
                "Number of out labels does not match the number of out indices to be plotted."
        else:
            out_labels = out_index.copy()

        ranks = da.array([self.ranking(out_index, np.sort(np.random.choice(self.r, self.r, replace=True)), range_key)
                          for _ in range(n_samples)])
        if ranks.ndim == 2:
            ranks = ranks[:, None, :]

        ranks = ranks.compute()
        stats = np.quantile(ranks, 0.5, 0)

        ranks = np.moveaxis(np.apply_along_axis(np.bincount, 0, ranks, minlength=self.g + 1)[1:].T, 1, 0)
        ranks = ranks / n_samples

        is_multi = False
        if path:
            path = Path(path)
            if ranks.shape[0] > 1:
                path.mkdir(exist_ok=True, parents=True)
                is_multi = True

        extent = truncate_after if truncate_after else self.g

        figs = []
        for o, oname in enumerate(out_labels):
            fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
            fig: plt.Figure
            ax: plt.Axes

            order = np.argsort(stats[o], 0)

            matshow = ax.matshow(ranks[o, order][:truncate_after, :truncate_after],
                                 extent=[0.5, extent + 0.5, extent - 0.5, -0.5],
                                 cmap=cmap)
            ax.set_xlabel("Ranking")
            ax.set_ylabel("Factor")

            ax.xaxis.get_major_locator().set_params(integer=True)

            ax.set_yticks(range(self.g)[:truncate_after])
            if factor_labels:
                ax.set_yticklabels([factor_labels[i] for i in order][:truncate_after])
            else:
                ax.set_yticklabels(np.arange(self.g)[order][:truncate_after])
            for t in ax.yaxis.get_major_ticks()[1::2]:
                t.set_pad(20)

            name = oname if not isinstance(oname, int) else f'{oname:03}'
            ax.set_title(name + f"\n(Number of resamples: {n_samples})" + f"\n(Using {range_key} points)")

            fig.colorbar(matshow, ax=ax)

            fig.tight_layout()
            if path:
                fig.savefig(path / name if is_multi else path, transparent=False, facecolor='white')

            plt.close()
            figs.append(fig)

        if len(figs) == 1:
            return figs[0]
        return figs

    @pass_or_compute
    def _calculate_ee(self,
                      out_index: List[int],
                      traj_index: List[int],
                      range_key: str) -> da.Array:
        """ Returns an array of the Estimated Effects themselves.

        Parameters
        ----------
        Inherited, out_index traj_index
            Parameters have the same meaning as in :meth:`__getitem__` but only processed lists of indices are accepted.
        range_key
            :code:`'all'`, :code:`'short'` or :code:`'long'` indicating which trajectory points to use.

        Returns
        -------
        numpy.ndarray
            :math:`\\hat{r} \\times \\hat{g} \\times \\hat{h}` array of elementary effects for every factor/group using
            the trajectories, points and outputs selected (:math:`\\hat{r}`, :math:`\\hat{g}` and :math:`\\hat{h}`
            respectively).
        """
        if not self.has_short_range:
            if range_key == 'all':
                range_key = 'long'
            else:
                raise ValueError('Range key can only be specified if trajectories include short range analysis points.')

        mid = self.g + 1
        end = 2 * self.g + 1
        comp_indices = {'all': np.arange(1, end),
                        'short': np.arange(mid, end),
                        'long': np.arange(1, mid)}[range_key]

        x_diffs = da.subtract(self.trajectories[traj_index][:, None, 0],
                              self.trajectories[traj_index][:, comp_indices])
        y_diffs = da.subtract(self.outputs[traj_index][:, [0]][:, :, out_index],
                              self.outputs[traj_index][:, comp_indices][:, :, out_index])

        if not self.is_grouped:
            x_diffs = da.sum(x_diffs, axis=2)
        else:
            x_diffs = da.sqrt(da.sum(x_diffs ** 2, axis=2))

        ee = y_diffs / x_diffs[:, :, None]

        return ee

    @pass_or_compute
    def _calculate_metrics(self,
                           out_index: List[int],
                           traj_index: List[int],
                           range_key: str) -> da.Array:
        """ Calculates the Estimated Effects metrics (:math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`).

        Parameters
        ----------
        Inherited, out_index traj_index
            Parameters have the same meaning as in :meth:`__getitem__` but only processed lists of indices are accepted.
        range_key
            :code:`'all'`, :code:`'short'` or :code:`'long'` indicating which trajectory points to use.

        Returns
        -------
        numpy.ndarray
            Three dimensional array of selected metrics, factors/groups and outputs.
            Metrics are ordered: :math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`.
        """
        ee = self._calculate_ee(out_index, traj_index, range_key)

        mu = da.mean(ee, axis=0)
        mu_star = da.mean(da.absolute(ee), axis=0)
        sigma = da.std(ee, axis=0, ddof=1) if ee.shape[0] > 1 else da.full_like(mu, np.nan)

        return da.array([mu, mu_star, sigma])

    def _plotting_core(self, path: Union[None, Path, str],
                       out_index: SpecialSlice,
                       plot_stub: Callable[..., plt.Axes],
                       **kwargs) -> Union[plt.Figure, List[plt.Figure]]:
        """ Provides common looping and saving infrastructure for plotting routines which produce one image per output.
        """
        out_index = self._expand_index('output', out_index)

        is_multi = False
        if path:
            path = Path(path)
            if len(out_index) > 1:
                path.mkdir(exist_ok=True, parents=True)
                is_multi = True

        is_custom_out_labs = 'out_labels' in kwargs and kwargs['out_labels'] is not None
        if is_custom_out_labs:
            if isinstance(kwargs['out_labels'], str):
                kwargs['out_labels'] = [kwargs['out_labels']]
            assert len(out_index) == len(kwargs['out_labels']), \
                "Number of out labels does not match the number of out indices to be plotted."

        is_custom_fact_labs = 'factor_labels' in kwargs and kwargs['factor_labels'] is not None
        if is_custom_fact_labs:
            if isinstance(kwargs['factor_labels'], str):
                kwargs['factor_labels'] = [kwargs['factor_labels']]
            assert self.g == len(kwargs['factor_labels']), \
                "Number of factor labels does not match the number of factor indices to be plotted."

        figs = []
        for i, index in enumerate(out_index):
            fig = plt.figure()
            figs.append(fig)

            ax = plot_stub(fig, index, **kwargs)

            name = f'{index:03}' if isinstance(index, (float, int)) else index
            if is_custom_out_labs:
                name = kwargs['out_labels'][i]
                ax.set_title(ax.get_title() + f"\n({name})")

            fig.tight_layout()
            if path:
                fig.savefig(path / name if is_multi else path, transparent=False, facecolor='white')

            plt.close()

        if len(figs) == 1:
            return figs[0]
        return figs

    def _plot_sensitivities_stub(self, fig: plt.Figure,
                                 out_index: int,
                                 range_key: Union[str, Sequence[str]],
                                 factor_labels: Optional[Sequence[str]] = None,
                                 log_scale: bool = False,
                                 **kwargs) -> plt.Axes:
        if isinstance(range_key, str):
            range_key = [range_key]
        n_rows = len(range_key)

        fig.set_size_inches(WIDTH, HEIGHT * n_rows)
        cmap = colors.ListedColormap(COLORS)

        hidden_ax = []
        nice_titles = {'short': "Using only short-range parameter changes",
                       'long': "Using only long-range parameter changes",
                       'all': "Using all trajectory points"}

        # Add Figure title
        axt: plt.Axes = fig.add_subplot(111)
        hidden_ax.append(axt)
        axt.set_title("Sensitivity classification of input factors" +
                      ("(Factors with $\\sigma=\\mu^*=0$ excluded.)" if log_scale else "") +
                      "\n" + nice_titles[range_key[0]])

        leg = []
        axes = []
        lims = [[float('inf'), -float('inf'), float('inf'), -float('inf')],
                [float('inf'), -float('inf'), float('inf'), -float('inf')]]  # Widest limits for two plot types
        for n_row, row_key in enumerate(range_key):
            if n_row >= 1:
                axh: plt.Axes = fig.add_subplot(n_rows, 1, n_row + 1)
                hidden_ax.append(axh)
                axh.set_title(nice_titles[row_key])

            ax0: plt.Axes = fig.add_subplot(n_rows, 2, 2 * n_row + 1)
            ax1: plt.Axes = fig.add_subplot(n_rows, 2, 2 * n_row + 2)
            axes += [ax0, ax1]

            ax0.set_ylabel("$\\sigma$", fontsize=int(1.5 * FONTSIZE))
            ax1.set_ylabel("$\\sigma/\\mu^*$", fontsize=int(1.5 * FONTSIZE))

            # Get metrics
            metrics = self[1:, :, out_index, :, row_key].compute()
            mu_star = metrics[0]
            sigma = metrics[1]

            labs = factor_labels if factor_labels is not None else np.arange(self.g)
            for j, ax in enumerate((ax0, ax1)):
                y = sigma.copy()
                if ax is ax1:
                    np.divide(sigma, mu_star, out=y, where=sigma != 0)

                ax.scatter(mu_star, y, marker='x', color='black', s=2)

                for k, lab in enumerate(labs):
                    ax.annotate(lab, (mu_star[k], y[k]), fontsize=FONTSIZE / 1.5)

                raw_xlims = ax.get_xlim()
                raw_ylims = ax.get_ylim()

                if log_scale:
                    if not np.any(mu_star > 0):
                        warnings.warn(
                            "Data has no positive values and cannot be log-scaled. Usually occurs when an output"
                            "is totally insensitive to all factor changes.", RuntimeWarning)
                        break
                    ax.set_xscale('log', nonpositive='mask')
                    ax.set_yscale('log', nonpositive='mask')

                    raw_xlims = 0.9 * mu_star[mu_star > 0].min(), ax.get_xlim()[1]
                    raw_ylims = 0.9 * y[y > 0].min(), ax.get_ylim()[1]

                lims[j][0] = min(lims[j][0], raw_xlims[0])
                lims[j][1] = max(lims[j][1], raw_xlims[1])
                lims[j][2] = min(lims[j][2], raw_ylims[0])
                lims[j][3] = max(lims[j][3], raw_ylims[1])

        for n_row, row_key in enumerate(range_key):  # Second loop so axis limits can be shared by all ranges.
            ax0, ax1 = axes[2 * n_row: 2 * n_row + 2]
            for j, ax in enumerate((ax0, ax1)):
                # Linear / Nonlinear Effect Line
                xlims = lims[j][:2]
                y_pts = []
                grads = np.array((0.1, 0.5, 1.0, np.sqrt(self.r) / 2))
                for i, grad in enumerate(grads):
                    ys = np.full(2, grad)
                    if ax is ax0:
                        ys *= np.array(xlims)
                    if i < 3:
                        ax.plot(xlims, ys,
                                color='black',
                                linewidth=0.5,
                                linestyle='solid',
                                zorder=1000)
                        y_pts.append(ys[1])
                    else:
                        ax.plot(xlims, ys,
                                color='black',
                                linewidth=0.7,
                                linestyle='dotted',
                                zorder=1000)

                # Influential / Non-influential Line
                ylims = lims[j][2:]
                if self.ct > xlims[0]:
                    ax.vlines(self.ct, ylims[0], ylims[1], color='black', linewidth=0.5)
                    ni = patches.Polygon([[xlims[0], ylims[0]],
                                          [xlims[0], ylims[1]],
                                          [self.ct, ylims[1]],
                                          [self.ct, ylims[0]]],
                                         facecolor='black',
                                         edgecolor='black',
                                         alpha=0.3,
                                         zorder=-1000)
                    ax.add_patch(ni)

                # Patches
                x_min = max(self.ct, xlims[0])
                y_min = np.array(grads)
                if ax is ax0:
                    y_min *= x_min
                y_min = np.concatenate(([ylims[0]], y_min[:-1], [ylims[1]]))
                y_pts = np.concatenate(([ylims[0]], y_pts, [ylims[1]]))
                for i, l in enumerate(('Linear', 'Monotonic', 'Quasi-Monotonic', 'Interacting')):
                    poly = patches.Polygon([[x_min, y_min[i]],
                                            [xlims[1], y_pts[i]],
                                            [xlims[1], y_pts[i + 1]],
                                            [x_min, y_min[i + 1]]],
                                           facecolor=cmap.colors[i],
                                           edgecolor=cmap.colors[i],
                                           zorder=-1000)
                    ax.add_patch(poly)
                    if n_row == 0 and ax is ax0:
                        leg.append(patches.Patch(cmap.colors[i], cmap.colors[i], label=l))

                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)

        leg.append(patches.Patch('black', 'black', alpha=0.3, label='Non-Influential'))
        leg.append(lines.Line2D([], [], c='black', ls='dotted', lw=0.7, label='SEM'))
        fig.legend(handles=leg, loc='lower center', ncol=6)

        # Hide title Axes
        for ax in hidden_ax:
            ax.set_xlabel("$\\mu^*$", fontsize=int(1.5 * FONTSIZE))
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        return axt

    def _plot_single_ranking_stub(self, fig: plt.Figure,
                                  out_index: int,
                                  range_key: str,
                                  factor_labels: Optional[Sequence[str]] = None,
                                  log_scale: bool = False,
                                  **kwargs) -> plt.Axes:
        """ Makes the rankings bar plot when one range key is requested. """
        fig.set_size_inches(WIDTH, (0.2 * self.g + 2) / 7 * HEIGHT)
        ax = fig.add_subplot(111)

        truncate_after = kwargs['truncate_after']

        ax.set_title("Parameter Ranking")
        ax.set_xlabel("$\\mu^*$", fontsize=int(1.5 * FONTSIZE))

        mu_star = self[1, :, out_index, :, range_key].squeeze().compute()
        i_sort = np.argsort(mu_star)
        if factor_labels is None:
            labs = i_sort.astype(str)
        else:
            labs = np.array(factor_labels)[i_sort]
        ax.barh(range(i_sort.size)[:truncate_after], mu_star[i_sort][:truncate_after])

        ax.set_yticks(range(i_sort.size)[:truncate_after])
        ax.set_yticklabels(labs[:truncate_after])

        ax.set_xscale('log' if log_scale else 'linear')

        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.margins(y=0)

        return ax

    def _plot_double_ranking_stub(self, fig: plt.Figure,
                                  out_index: int,
                                  range_key: Sequence[str],
                                  factor_labels: Optional[Sequence[str]] = None,
                                  log_scale: bool = False,
                                  **kwargs) -> plt.Axes:
        """ Makes the :math:`\\mu^*` v :math:`\\mu^*` scatter ranking plot when two range keys are requested. """
        fig.set_size_inches(WIDTH, WIDTH)
        ax: plt.Axes = fig.add_subplot(111)

        ax.set_title("Parameter Ranking")
        ax.set_xlabel(rf"$\mu^*$ (Using {range_key[0] + ('-range' if range_key[0] != 'all' else '')} points)")
        ax.set_ylabel(rf"$\mu^*$ (Using {range_key[1] + ('-range' if range_key[1] != 'all' else '')} points)")
        ax.axline((0, 0), slope=1, color='gray', linewidth=0.8, zorder=-500)

        first = self[1, :, out_index, :, range_key[0]].squeeze().compute()
        second = self[1, :, out_index, :, range_key[1]].squeeze().compute()
        ax.scatter(first, second, marker='x', s=2)

        labs = np.arange(self.g) if not factor_labels else factor_labels
        for k, lab in enumerate(labs):
            ax.annotate(lab, (first[k], second[k]), fontsize=FONTSIZE // 1.5)

        ax.set_yscale('log' if log_scale else 'linear')
        ax.set_xscale('log' if log_scale else 'linear')
        ax.set_aspect('equal', 'datalim')

        return ax

    def _expand_index(self, index_type: str, index) -> List[int]:
        """ Converts the various ways to specify the output/group index into a list of absolute index positions. """
        full = {'metric': 3,
                'factor': self.g,
                'output': self.h,
                'trajec': self.r}[index_type]

        key_map = {'mu': 0,
                   'mu_star': 1,
                   'sigma': 2}

        if isinstance(index, np.ndarray):
            index = getattr(index, "tolist", lambda: index)()

        if index is None or index == 'all' or index == slice(None):
            return list(range(full))

        if isinstance(index, slice):
            return list(range(full))[index]

        if isinstance(index, str) or isinstance(index, int):
            index = [index]

        proc_index = []
        for i in index:
            if i == 'all':
                [proc_index.append(_) for _ in range(full)]
            elif isinstance(i, str) and index_type == 'metric':
                proc_index.append(key_map[i])
            elif isinstance(i, int) or (i == 'mean' and index_type == 'output'):
                proc_index.append(i)
            else:
                raise ValueError(f"Cannot parse {i} in index.")

        return proc_index
