import copy
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ..common.wrappers import needs_optional_package

try:
    import matplotlib.pyplot as plt
    from matplotlib import lines
    from matplotlib import patches
    from matplotlib import cm

    plt.rcParams['font.size'] = 8
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['savefig.format'] = 'svg'
    plt.rcParams['figure.figsize'] = 6.7, 3.35
    plt.rcParams['font.size'] = 6
except (ModuleNotFoundError, ImportError, TypeError):  # TypeError caught for building docs
    pass

__all__ = ('EstimatedEffects',)

SpecialSlice = Union[None, int, str, List, slice]


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
           in this regime because it is not possible to define . Only :math:`\\mu^*` is accessible.

    convergence_threshold
        See :attr:`convergence_threshold`.

    cutoff_threshold
        See :attr:`ct`.

    trajectory_style
        See :attr:`traj_style`

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

    out_dims : int
        Dimensionality of the output if one would like to investigate factor sensitivities against multiple function
        responses. Often referred to as :math:`h` in the documentation of equations. See :attr:`h`.

    outputs : numpy.ndarray
        :math:`r \\times (g+1) \\times h` array of function evaluations corresponding to the input factors in
        :attr:`trajectories`. Represents the responses of the model to the points in the :attr:`trajectories`.

    trajectories : numpy.ndarray
        :math:`r \\times (g+1) \\times k` array of :math:`r` trajectories, each with :math:`g+1` points of :math:`k`
        factors each. Represents the carefully sampled points in the factor space where the model will be evaluated.

    traj_style : str
        The style of the trajectories which will be used in this calculation. Accepts :code:`'radial'` and
        :code:`'stairs'`. See :mod:`.trajectories`.
    """

    @property
    def mu(self):
        """ Shortcut access to the Estimated Effects :math:`\\mu` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to: :code:`ee['mu', :, 'mean', :]`

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
    def mu_star(self):
        """ Shortcut access to the Estimated Effects :math:`\\mu^*` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to:
        :code:`ee['mu_star', :, 'mean', :]`

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
    def sigma(self):
        """ Shortcut access to the Estimated Effects :meth:`\\sigma` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to: :code:`ee['sigma', :, 'mean', :]`

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
    def is_converged(self) -> np.ndarray:
        """ Converged if the instance has enough trajectories for the factor ordering to be stable.
        Returns :obj:`True` if the change in :meth:`position_factor` over the last 10 trajectory entries is smaller
        than :attr:`convergence_threshold`.

        Returns
        -------
        numpy.ndarray
            Array of length :math:`h` with boolean values indicating if the sensitivity metrics for that output have
            converged.
        """
        if self.r < 10:
            return np.full(self.h, False)
        return np.squeeze(np.abs(self.position_factor(self.r - 10, self.r, 'all')) < self.convergence_threshold)

    def __init__(self, input_dims: int,
                 output_dims: int,
                 groupings: Optional[np.ndarray] = None,
                 convergence_threshold: float = 0,
                 cutoff_threshold: float = 0.1,
                 trajectory_style: str = 'radial'):
        self.logger = logging.getLogger('glompo.analyzer')

        self.trajectories = np.array([])
        self.dims: int = input_dims
        self.traj_style = trajectory_style
        self._groupings = groupings if groupings is not None else np.identity(input_dims)
        self._is_grouped = groupings is not None
        if np.sum(self._groupings) != input_dims:
            raise ValueError("Invalid grouping matrix, each factor must be in exactly 1 group.")

        self.outputs = np.array([])
        self.out_dims = output_dims

        self.convergence_threshold = convergence_threshold
        self.ct = cutoff_threshold
        self._metrics = np.array([[[]]])

    def __getitem__(self, item):
        """ Retrieves the sensitivity metrics (:math:`\\mu, \\mu^*, \\sigma`) for a particular calculation configuration
        The indexing is a strictly ordered set of maximum length 4:

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
            calculation. Accepts one of the following:

               * Any integers or slices of 0, 1, ... :math:`h`: The metrics for the outputs at the
                 corresponding index will be used.

               * :code:`'mean'`: The metrics will be averaged across the output dimension.

        * Fourth Index (:code:`traj_index`):
           Which trajectories to use in the calculation.

        Parameters
        ----------
        item
            Tuple of slices and indices. Maximum length of 4.

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

        """
        if self.trajectories.size == 0:
            warnings.warn("Please add at least one trajectory before attempting to access calculations.", UserWarning)
            return np.array([[[]]])

        if not isinstance(item, tuple):
            item = (item,)  # NOT the same as tuple(item) since you don't want to expand 'mu' to ('m', 'u')
        item = [slice(None) if i is None or i == 'all' else i for i in item]
        item += [slice(None)] * (4 - len(item))
        m, k, h, t = item

        if not isinstance(m, slice):
            m = np.atleast_1d(m)
            for s, i in (('mu', 0), ('mu_star', 1), ('sigma', 2)):
                m[[_ == s for _ in m]] = i
            m = m.astype(int)

        if self.is_grouped and (isinstance(m, slice) or any([i in m for i in [0, 2]])):
            raise ValueError('Cannot access mu and sigma metrics if groups are being used')

        orig_h = copy.copy(h)
        nest_out = isinstance(h, list) or isinstance(h, tuple)
        mean_out = h == 'mean' or (nest_out and 'mean' in h)
        h = slice(None) if mean_out else h

        # Attempt to access existing results
        if t == slice(None) and self._metrics.size > 0:
            metrics = self._metrics
        else:
            metrics = self._calculate(k, t)
            if k == slice(None) and t == slice(None):
                # Save results to cache if a full calculation was done.
                self._metrics = metrics

        metrics = metrics[m, k, h]

        if nest_out:
            cols = []
            for o in orig_h:
                if o == 'all':
                    cols += np.split(metrics, self.h, axis=2)
                elif o == 'mean':
                    cols.append(np.mean(metrics, 2)[:, :, None])
                else:
                    cols.append(metrics[:, :, o, None])
            metrics = np.concatenate(cols, axis=2)

        if mean_out and not nest_out:
            return np.mean(metrics, 2)

        return metrics

    def add_trajectory(self, trajectory: np.ndarray, outputs: np.ndarray):
        """ Add a trajectory of points and their corresponding model output to the calculation.

        Parameters
        ----------
        trajectory
            A trajectory of points as produced by one of the trajectory generation functions (see :mod:`.trajectories`).
            Should have a shape of :math:`(k+1) \\times k` where :math:`k` is the number of factors / dimensions of the
            input.
        outputs
            :math:`(k+1) \\times h` model outputs corresponding to the points in the `trajectory`. Where :math:`h` is
            the dimensionality of the outputs.

        Raises
        ------
        ValueError
            If `trajectory` or `outputs` do not match the dimensions above.

        Notes
        -----
        The actual calculation of the Estimated Effects metrics is not performed in this method. Add new trajectories is
        essentially free. The calculation is only performed the moment the user attempts to access any of the metrics.
        The results of the calculation are held in memory, thus if the number of trajectories remains unchanged, the
        user may continue accessing the metrics at no further cost.
        """
        # Clear old results
        self._metrics = np.array([[[]]])

        if trajectory.shape != (self.g + 1, self.k):
            raise ValueError(f"Cannot parse trajectory with shape {trajectory.shape}, must be ({self.g + 1}, "
                             f"{self.k}).")
        if self.h > 1 and outputs.shape != (self.g + 1, self.h):
            raise ValueError(f"Cannot parse outputs with shape {outputs.shape}, must be ({self.g + 1}, {self.h})")

        if self.h == 1 and outputs.shape != (self.g + 1, self.h) and outputs.shape != (self.g + 1,):
            raise ValueError(f"Cannot parse outputs with shape {outputs.shape}, ({self.g + 1},) expected.")

        if self.h == 1:
            outputs = outputs.reshape((self.g + 1, self.h))

        if len(self.trajectories) > 0:
            self.trajectories = np.append(self.trajectories, [trajectory], axis=0)
            self.outputs = np.append(self.outputs, [outputs], axis=0)
        else:
            self.trajectories = np.array([trajectory])
            self.outputs = np.array([outputs])

    def generate_add_trajectory(self, style: Optional[None]):
        """ Convenience method to automatically generate a trajectory and add it to the calculation. """
        # TODO Implement

    def build_until_convergence(self,
                                func: Callable[[np.ndarray], float],
                                r_max: int):
        """  """
        # TODO Implement
        raise NotImplementedError

    def position_factor(self, i: int, j: int, out_index: SpecialSlice = 'mean') -> np.ndarray:
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
        out_index
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
        pos_i = np.atleast_2d(self.ranking(out_index, slice(None, i + 1)))
        pos_j = np.atleast_2d(self.ranking(out_index, slice(None, j + 1)))
        return np.sum(2 * np.abs(pos_i - pos_j) / (pos_i + pos_j), axis=1).squeeze()

    def order_factors(self,
                      out_index: SpecialSlice = 'mean',
                      traj_index: SpecialSlice = None) -> np.ndarray:
        """ Returns factor indices in descending order of their influence on the outputs.
        The *positions* in the array are the rankings, the *contents* of the array are the factor / group indices. This
        is the inverse of :meth:`ranking`.

        Parameters
        ----------
        See :meth:`__getitem__`.

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
        metric = np.atleast_3d(self[1, :, out_index, traj_index])
        return metric.argsort(1)[:, -1::-1].squeeze().T

    def ranking(self, out_index: SpecialSlice = 'mean', traj_index: SpecialSlice = None) -> np.ndarray:
        """ Returns the ranking of each factor being analyzed.
        The *positions* in the array are the factor or group indices, the *contents* of the array are rankings such that
        1 is the most influential factor / group and :math:`g+1` is the least influential. This is the inverse of
        :meth:`order_factors`.

        Parameters
        ----------
        See :meth:`__getitem__`.

        Returns
        -------
        numpy.ndarray
            Array of :math:`g` length vectors for each of the selected outputs.

        See Also
        --------
        :meth:`order_factors`
        """
        return np.squeeze(np.atleast_2d(self.order_factors(out_index, traj_index)).argsort(1) + 1)

    def classification(self, n_cats: int, out_index: Union[int, str] = 'mean') -> Dict[str, np.ndarray]:
        """ Returns a dictionary with each factor index classified according to its effect on the function outputs.

        Parameters
        ----------
        n_cats
            Number of categories in the classification. Two categorizations from literature are supported:

            * :code:`n_cats = 3`:

              Three category classification of `Vanrolleghem et al (2015)
              <https://doi.org/10.1016/J.JHYDROL.2014.12.056>`_:

              ========================= ========================================================== ===
              Name                      Condition                                                  Description
              ========================= ========================================================== ===
              :code:`'non-influential'` :math:`\\mu^* < \\text{CT}`                                  No appreciable effect.
              :code:`'important'`       :math:`\\mu^* > \\text{CT} \\& \\sigma/\\mu^* < \\sqrt{r}/2`     Strong individual effect.
              :code:`'interacting'`     :math:`\\mu^* > \\text{CT} \\& \\sigma/\\mu^* > \\sqrt{r}/2`     Strong interacting effects.
              ========================= ========================================================== ===

            * :code:`n_cats = 5`:

              Five category classification of `Garcia Sanchez et al. (2014)
              <https://doi.org/10.1016/J.ENBUILD.2012.08.048>`_:

              ========================= ================================================== ===
              Name                      Condition                                          Description
              ========================= ================================================== ===
              :code:`'non-influential'` :math:`\\mu^* < \\text{CT}`                          No appreciable effect.
              :code:`'linear'`          :math:`\\mu^* > \\text{CT} \\& \\sigma/\\mu^* < 0.1`    Strong linear effect.
              :code:`'monotonic'`       :math:`\\mu^* > \\text{CT} \\& \\sigma/\\mu^* < 0.2`    Strong monotonic effect.
              :code:`'quasi-monotonic'` :math:`\\mu^* > \\text{CT} \\& \\sigma/\\mu^* < 0.5`    Moderate monotonic effect.
              :code:`'interacting'`     :math:`\\mu^* > \\text{CT} \\& \\sigma/\\mu^* < 1.0`    Strongly non-linear effects.
              ========================= ================================================== ===

        out_index
            Output dimension along which to do the classification.

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
        generated at a time.

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
        mu, ms, sd = self[:, :, out_index, :]

        fr = np.zeros_like(mu)
        np.divide(sd, ms, out=fr, where=sd != 0)

        if n_cats == 3:
            grad = np.sqrt(self.r) / 2
            return {'important': np.argwhere((ms > self.ct) & (fr < grad)).ravel(),
                    'interacting': np.argwhere((ms > self.ct) & (fr >= grad)).ravel(),
                    'non-influential': np.argwhere(ms < self.ct).ravel()}

        if n_cats == 5:
            return {'non-influential': np.argwhere(ms < self.ct).ravel(),
                    'linear': np.argwhere((ms > self.ct) & (fr <= 0.1)).ravel(),
                    'monotonic': np.argwhere((ms > self.ct) & (fr > 0.1) & (fr <= 0.5)).ravel(),
                    'quasi-monotonic': np.argwhere((ms > self.ct) & (fr > 0.5) & (fr <= 1.0)).ravel(),
                    'interacting': np.argwhere((ms > self.ct) & (fr > 1)).ravel(),
                    }

        raise ValueError(f'Cannot parse n_cats = {n_cats}, must equal 3 or 5.')

    def relevance_factor(self, category: str, out_index: Union[int, str] = 'mean') -> np.ndarray:
        """ Measure of the fraction of factors which have an influence on the outputs.
        Calculated as:

        .. math::

           Rel_{\\text{category}} = \\frac{N_\\text{category}}{g}

        Where :math:`N_\\text{category}` are the number of factors classified into a particular classification category.

        Parameters
        ----------
        category
            Classification key (see :meth:`classification`) for
        out_index
            See :meth:`__getitem__`

        References
        ----------
        Cosenza, A., Mannina, G., Vanrolleghem, P. A., & Neumann, M. B. (2013). Global sensitivity analysis in
        wastewater applications: A comprehensive comparison of different methods. Environmental Modelling & Software,
        49, 40–52. https://doi.org/10.1016/J.ENVSOFT.2013.07.009
        """
        return self.classification(out_index)[category] / self.k

    def bootstrap_metrics(self,
                          metric_index: SpecialSlice = None,
                          factor_index: SpecialSlice = None,
                          out_index: SpecialSlice = None):
        """  """

    @needs_optional_package('matplotlib')
    def plot_sensitivities(self,
                           path: Union[Path, str] = 'sensitivities',
                           out_index: SpecialSlice = 'mean',
                           factor_labels: Optional[Sequence[str]] = None,
                           log_scale: bool = False):
        """ Saves a sensitivity plot.
        Produces two side-by-side scatter plots. The first is :math:`\\mu^*` versus :math:`\\sigma`, the second is
        :math:`\\mu^*` versus :math:`\\sigma/\\mu^*`. Defined dividers are included to classify factors into:

           #. 'important'
           #. 'linear'
           #. 'monotonic'
           #. 'interacting'
           #. 'non-influential'

        Parameters
        ----------
        path
            If one plot is being produced (see `out_index`) this is interpreted as the filename with which to save the
            figure. If multiple plots are produced for all the outputs then this is interpreted as a directory into
            which the figures will be saved.
        out_index
            See :meth:`__getitem__`. If :obj:`None` or :code:`'all'`, one plot will be created for each output.
        factor_labels
            Optional sequence of descriptive names for each factor to add to the figure. Defaults to the factor's
            index position.
        log_scale
            If :obj:`True` the axes will be plotted on a log scale.

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
        self._plotting_core(path, out_index, self._plot_sensitivities_stub,
                            factor_labels=factor_labels, log_scale=log_scale)

    @needs_optional_package('matplotlib')
    def plot_rankings(self,
                      path: Union[Path, str] = 'ranking',
                      out_index: SpecialSlice = 'mean',
                      factor_labels: Optional[Sequence[str]] = None):
        """ Saves the factor rankings as a plot.
        Plots the ordered :math:`\\mu^*` values against their corresponding parameter indices.

        Parameters
        ----------
        See :meth:`plot_sensitivities`.
        """
        self._plotting_core(path, out_index, self._plot_ranking_stub, factor_labels=factor_labels)

    @needs_optional_package('matplotlib')
    def plot_convergence(self,
                         path: Union[Path, str] = 'sensi_convergence',
                         out_index: SpecialSlice = 'mean',
                         step_size: int = 10,
                         out_labels: Optional[Sequence[str]] = None):
        """ Plots the evolution of the Position Factor ($PF_{r_i \\to r_j}$) metric as a function of increasing
        number of trajectories.

        Parameters
        ----------
        path
            Path to file into which the plot should be saved.
        out_index
            See :meth:`__getitem__`. If multiple output dimensions are selected, they will be included on the same plot
        step_size
            The step size in number of trajectories when calculating the position factor.
        out_labels
            Optional sequence of descriptive labels for the plot legend corresponding to the outputs selected to be
            plotted. Defaults to 'Output 0', 'Output 1', 'Output 2', ...

        Notes
        -----
        The Position Factor metric is a measure of how much rankings have changed between the rankings calculated
        using :math:`i` trajectories, and the rankings calculated using :math:`j` trajectories sometime later. Thus, if
        `step_size` is 10 then the plot would show the evolution of the Position Factor at:
        :math:`0 \\to 10, 10 \\to 20, 20 \\to 30, ...`

        See Also
        --------
        :meth:`position_factor`
        """
        steps = np.clip([(i, i + step_size) for i in range(1, self.r, step_size)], None, self.r)
        pf = np.array([np.atleast_1d(self.position_factor(*pair, out_index)) for pair in steps])

        fig, ax = plt.subplots()
        fig: plt.Figure
        ax: plt.Axes

        ax.set_title('Convergence of sensitivity rankings ($PF$ v $r$)')
        ax.set_xlabel('Trajectories Compared ($i \\to j$)')
        ax.set_ylabel('Position Factor ($PF_{i \\to j}$)')

        ax.plot(pf, marker='.')
        if pf.shape[1] > 1:
            labs = out_labels if out_labels is not None else [f'Output {i}' for i in range(pf.shape[1])]
            ax.legend(labels=labs)

        ax.set_xticks([i for i, _ in enumerate(steps)])
        ax.set_xticklabels([f'{s[0]}$\\to${s[1]}' for s in steps], rotation=45)

        fig.tight_layout()
        fig.savefig(path)

        plt.close(fig)

    def invert(self):
        """  """
        # TODO Implement

    def _calculate(self,
                   out_index: SpecialSlice,
                   traj_index: Union[int, slice, List[int]]) -> np.ndarray:
        """ Calculates the Estimated Effects metrics (:math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`).

        Parameters
        ----------
        out_index
            See :meth:`__getitem__`.
        traj_index
            See :meth:`__getitem__`.

        Returns
        -------
        numpy.ndarray
            Three dimensional array of selected metrics, factors/groups and outputs.
            Metrics are ordered: :math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`.

        Notes
        -----
        If `traj_index` is provided then the results will *not* be saved as the :attr:`mu`, :attr:`mu_star` and :
        attr:`sigma` attributes of the class.

        If all available trajectories are used *and* all metrics for all the outputs are calculated, then the results
        are saved into the above mentioned attributes.

        If a user attempts to access any of the attributes (and the number of trajectories has changed since the last
        call), this method is automatically called for all available trajectories. In other words, there is never any
        risk of accessing metrics which are out-of-sync with the number of trajectories appended to the calculation.
        """
        if self.traj_style == 'stairs':
            x_diffs = self.trajectories[traj_index, :-1] - self.trajectories[traj_index, 1:]
            y_diffs = self.outputs[traj_index, :-1, out_index] - self.outputs[traj_index, 1:, out_index]

        else:  # Radial style trajectories
            x_diffs = self.trajectories[traj_index, 0] - self.trajectories[traj_index, 1:]
            y_diffs = self.outputs[traj_index, 0, out_index] - self.outputs[traj_index, 1:, out_index]

        where = np.where(x_diffs @ self.groupings)[0::2]
        if not self.is_grouped:
            x_diffs = np.sum(x_diffs, axis=2)
        else:
            x_diffs = np.sqrt(np.sum(x_diffs ** 2, axis=2))

        ee = y_diffs / x_diffs[:, :, None]
        ee[where] = ee.copy().ravel().reshape(-1, self.h)

        mu = np.mean(ee, axis=0)
        mu_star = np.mean(np.abs(ee), axis=0)
        sigma = np.std(ee, axis=0, ddof=1) if ee.shape[0] > 1 else np.full_like(mu, np.nan)

        return np.array([mu, mu_star, sigma])

    def _plotting_core(self, path: Union[Path, str], out_index: SpecialSlice, plot_stub: Callable[..., None], **kwargs):
        """ Most plot function require the same looping and gathering of metrics, this is done here and then passed
        to the `plot_stub` method which have the individualized plot commands.
        """
        if not self.is_grouped:
            metrics = np.atleast_3d(self[:, :, out_index])
        else:
            metrics = np.atleast_3d(self[['mu_star'], :, out_index])

        path = Path(path)
        is_multi = False
        if metrics.shape[2] > 1:
            path.mkdir(exist_ok=True, parents=True)
            is_multi = True

        for i in range(metrics.shape[2]):
            fig = plt.figure()

            if self.is_grouped:
                plot_stub(fig, mu_star=metrics[0, :, i], **kwargs)
            else:
                plot_stub(fig, mu=metrics[0, :, i], mu_star=metrics[1, :, i], sigma=metrics[2, :, i], **kwargs)

            fig.tight_layout()
            fig.savefig(path / f'{i:03}' if is_multi else path)
            plt.close(fig)

    def _plot_sensitivities_stub(self, fig: plt.Figure,
                                 mu: Optional[np.ndarray] = None,
                                 mu_star: Optional[np.ndarray] = None,
                                 sigma: Optional[np.ndarray] = None,
                                 factor_labels: Optional[Sequence[str]] = None,
                                 log_scale: bool = False):
        fig.set_size_inches(6.7, 3.35)
        cmap = cm.get_cmap('Pastel2')

        axt: plt.Axes = fig.add_subplot(111)
        ax0: plt.Axes = fig.add_subplot(121)
        ax1: plt.Axes = fig.add_subplot(122)

        ax0.set_ylabel("$\\sigma$")
        ax1.set_ylabel("$\\sigma/\\mu^*$")

        axt.set_title("Sensitivity classification of input factors" +
                      ("(Factors with $\\sigma=\\mu^*=0$ excluded.)" if log_scale else ""))
        axt.set_xlabel("$\\mu^*$")
        axt.spines['top'].set_color('none')
        axt.spines['bottom'].set_color('none')
        axt.spines['left'].set_color('none')
        axt.spines['right'].set_color('none')
        axt.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        leg = []
        labs = factor_labels if factor_labels is not None else np.arange(self.g)
        for ax in (ax0, ax1):
            y = sigma.copy()
            if ax is ax1:
                np.divide(sigma, mu_star, out=y, where=sigma != 0)

            ax.scatter(mu_star, y, marker='x', color='black', s=2)

            for j, lab in enumerate(labs):
                ax.annotate(lab, (mu_star[j], y[j]), fontsize=5)

            raw_xlims = ax.get_xlim()
            raw_ylims = ax.get_ylim()

            if log_scale:
                ax.set_xscale('log', nonpositive='mask')
                ax.set_yscale('log', nonpositive='mask')

                raw_xlims = 0.9 * mu_star[mu_star > 0].min(), ax.get_xlim()[1]
                raw_ylims = 0.9 * y[y > 0].min(), ax.get_ylim()[1]

                ax.set_xlim(*raw_xlims)
                ax.set_ylim(*raw_ylims)

            # Linear / Nonlinear Effect Line
            xlims = ax.get_xlim()
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
            ylims = ax.get_ylim()
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
                if ax is ax0:
                    leg.append(patches.Patch(cmap.colors[i], cmap.colors[i], label=l))

            ax.set_xlim(*raw_xlims)
            ax.set_ylim(*raw_ylims)

        leg.append(patches.Patch('black', 'black', alpha=0.3, label='Non-Influential'))
        leg.append(lines.Line2D([], [], c='black', ls='dotted', lw=0.7, label='SEM'))
        fig.legend(handles=leg, loc='lower center', ncol=6)

    def _plot_ranking_stub(self, fig: plt.Figure,
                           mu: Optional[np.ndarray] = None,
                           mu_star: Optional[np.ndarray] = None,
                           sigma: Optional[np.ndarray] = None,
                           factor_labels: Optional[Sequence[str]] = None):
        fig.set_size_inches(6.7, 6.7)
        ax = fig.add_subplot(111)

        ax.set_title("Parameter Ranking")
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("$\\mu^*$")
        ax.tick_params(axis='x', rotation=90)

        i_sort = np.argsort(mu_star)
        if factor_labels is None:
            labs = i_sort.astype(str)
        else:
            labs = np.array(factor_labels)[i_sort]
        ax.bar(labs, mu_star[i_sort])
