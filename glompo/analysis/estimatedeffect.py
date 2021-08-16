import warnings
from pathlib import Path
from typing import Callable, Dict, List, Union

import numpy as np

from ..common.wrappers import needs_optional_package

try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):
    pass

__all__ = ('EstimatedEffects',)

# TODO Are groups compatible throughout?
# TODO Are multi-dim outputs compatible throughout?
# TODO Users will not know that to take all indices they must send slice(None) to out_index of methods

SpecialSlice = Union[int, str, List[int], List[str], slice]


class EstimatedEffects:
    # TODO Complete parameters
    # TODO Complete attributes
    """ Implementation of Morris screening strategy.
    Based on the original work of Morris (1991) but includes extensions published over the years.
    Global sensitivity method for expensive functions. Uses minimal number of function evaluations to develop a good
    proxy for the total sensitivity of each input factor. Produces three sensitivity measures (:math:`\\mu`,
    :math:`\\mu^*` and :math:`\\sigma`) that are able to capture magnitude and direction the sensitivity, as well as
    nonlinear and interaction effects. The user is directed to the references below for a detailed explanation of the
    meaning of each of these measures.

    Parameters
    ----------


    References
    ----------
    Morris, M. D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments. *Technometrics*, 33(2),
    161–174. https://doi.org/10.1080/00401706.1991.10484804

    Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large
    models. *Environmental Modelling & Software*, 22(10), 1509–1518. https://doi.org/10.1016/j.envsoft.2006.10.004

    Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M., & Tarantola, S. (2007).
    Global Sensitivity Analysis. The Primer (A Saltelli, M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli,
    M. Saisana, & S. Tarantola (eds.)). *John Wiley & Sons, Ltd.* https://doi.org/10.1002/9780470725184

    Ruano, M. V., Ribes, J., Seco, A., & Ferrer, J. (2012). An improved sampling strategy based on trajectory design for
    application of the Morris method to systems with many input factors. *Environmental Modelling & Software*, 37,
    103–109. https://doi.org/10.1016/j.envsoft.2012.03.008

    Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban water
    quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*, 522,
    339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056

    Attributes
    ----------
    dims: int
        The number of input factors for which sensitivity is being tested. Often referred to as :math:`k` throughout the
        documentation here to match literature.
    out_dims: int
        Dimensionality of the output if one would like to investigate factor sensitivities against multiple function
        responses. Often reffered to as :math:`h` in the documentation of equations.
    outputs : numpy.ndarray
        :math:`r \\times (k+1) \\times h` array of function evaluations corresponding to the input factors in
        :attr:`trajectories`. :math:`h` is the dimensionality of the outputs.
    """

    @property
    def mu(self):
        """ Shortcut access to the Estimated Effects :meth:`\\mu` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to: :code:`ee['mu', :, 'mean', :]`
        """
        return self['mu', :, 'mean', :].squeeze()

    @property
    def mu_star(self):
        """ Shortcut access to the Estimated Effects :meth:`\\mu^*` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to:
        :code:`ee['mu_star', :, 'mean', :]`
        """
        return self['mu_star', :, 'mean', :].squeeze()

    @property
    def sigma(self):
        """ Shortcut access to the Estimated Effects :meth:`\\sigma` metric using all trajectories, for all input
        dimensions, taking the average along output dimensions. Equivalent to: :code:`ee['sigma', :, 'mean', :]`
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
    def is_converged(self, out_index: SpecialSlice = 'mean') -> bool:
        """ Converged if the instance has enough trajectories for the factor ordering to be stable.
        Returns :obj:`True` if the change in :meth:`position_factor` over the last 10 trajectory entries is smaller
        than :attr:`convergence_threshold`.

        Parameters
        ----------
        out_index
            See :meth:`get_metrics`.
        """
        # TODO Rethink definition of converged
        return np.squeeze(self.position_factor(self.r - 10, self.r, out_index) < self.convergence_threshold)

    @property
    def classification(self, out_index: Union[int, str] = 'mean') -> Dict[str, np.ndarray]:
        """ Returns a dictionary with each factor index classified as :code:`'important'`, :code:`'interacting'` and
        :code:`'non-influential'`. Follows the definitive classification system of
        `Vanrolleghem et al. (2105) <https://doi.org/10.1016/J.JHYDROL.2014.12.056>`_.

        Categories are defined as follows:

        :code:`'important'`:
           These factors have a linear effect on the model output.

        :code:`'interacting'`:
           These factors have a nonlinear effect on the model output.

        :code:`'non-influential'`:
            These factors do not have a significant impact on model output.

        Parameters
        ----------
        out_index
            Output dimension along which to do the classification.

        References
        ----------
        Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban
        water quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*,
        522, 339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056
        """
        mu, ms, sd = self[:, :, out_index, :]
        return {'important': np.argwhere((ms > self.ct) & (sd < ms * np.sqrt(self.r) / 2)).ravel(),
                'interacting': np.argwhere((ms > self.ct) & (sd >= ms * np.sqrt(self.r) / 2)).ravel(),
                'non-influential': np.argwhere(ms < self.ct).ravel()}

    @property
    def ranking(self, out_index: SpecialSlice = 'mean') -> np.ndarray:
        """ Returns factor indices in descending order of their influence on the outputs.

        Parameters
        ----------
        out_index
            See :meth:`__get_item__`.
        """
        return np.squeeze(self[1, :, out_index].argsort()[:, -1::-1] + 1)

    def __init__(self, input_dims: int,
                 output_dims: int,
                 convergence_threshold: float = 0,
                 cutoff_threshold: float = 0.1,
                 trajectory_style: str = 'radial'):
        self._metrics = np.array([[[]]])

        self.trajectories = np.array([])
        self.dims: int = input_dims
        self.traj_style = trajectory_style

        self.outputs = np.array([])
        self.out_dims = output_dims

        self.convergence_threshold = convergence_threshold
        self.ct = cutoff_threshold

    def __getitem__(self, item):
        """ Retrieves the sensitivity metrics (:math:`\\mu, \\mu^*, \\sigma`) for a particular calculation configuration
        The indexing has a maximum length of 4:

        * First index:
            Indexes the metric. If :code:`:` is used, all metrics are returned.
            Also accepts strings which are paired to the following corresponding ints:
            === ===
            int str
            === ===
            0   'mu'
            1   'mu_star'
            2   'sigma'
            === ===

        * Second index:
           Indexes the factor. :code:`:` returns metrics for all factors.

        * Third index:
            Only applicable if a multidimensional output is being investigated. Determines which metrics to use in the
            calculation. Accepts one of the following:

               * Any integers or slices of 0, 1, ... :math:`h`: The metrics for the outputs at the corresponding index
                 will be used.

               * :code:`'mean'`: The metrics will be averaged across the output dimension.

               * :code:`:`: Metrics will be returned for all outputs.

        * Fourth Index:
           Which trajectories to use in the calculation. If not supplied or :code:`:`, all trajectories will be used.
           Accepts slices as well as lists of integers. Helpful for bootstrapping.

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

        Returns
        -------
        numpy.ndarray
            :math:`m \\times \\times k \\times h` array of sensitivity metrics where :math:`m` is the metric, :math:`k`
            is the factor and :math:`h` is the output dimensionality.
        """
        if self.trajectories.size == 0:
            warnings.warn("Please add at least one trajectory before attempting to access calculations.", UserWarning)
            return np.array([[[]]])

        m, h, k, t = (*item, *(slice(None),) * (4 - len(item)))

        if not isinstance(m, slice):
            m = np.atleast_1d(m)
            for s, i in (('mu', 0), ('mu_star', 1), ('sigma', 2)):
                m[m == s] = i
            m = m.astype(int)

        mean_out = k == 'mean'
        k = slice(None) if mean_out else k

        # Attempt to access existing results
        if t == slice(None) and self._metrics.size > 0:
            metrics = self._metrics
        else:
            metrics = self._calculate(k, t)
            if k == slice(None) and t == slice(None):
                # Save results to cache if a full calculation was done.
                self._metrics = metrics

        metrics = metrics[m, k, h]

        if mean_out:
            metrics = np.mean(metrics, 2)

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
        ValueErrror
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

        if trajectory.shape != (self.k + 1, self.k):
            raise ValueError(f"Cannot parse trajectory with shape {trajectory.shape}, must be ({self.k + 1}, "
                             f"{self.k}).")
        if self.h > 1 and outputs.shape != (self.k + 1, self.h):
            raise ValueError(f"Cannot parse outputs with shape {outputs.shape}, must be ({self.k + 1}, {self.h})")

        if self.h == 1 and (outputs.shape != (self.k + 1, self.h) or outputs.shape != (self.k + 1,)):
            raise ValueError(f"Cannot parse outputs with length {len(outputs)}, {self.k + 1} values expected.")

        if self.h == 1:
            outputs = outputs.reshape((self.k + 1, self.h))

        if len(self.trajectories) > 0:
            self.trajectories = np.append(self.trajectories, [trajectory], axis=0)
            self.outputs = np.append(self.outputs, [outputs], axis=0)
        else:
            self.trajectories = np.array([trajectory])
            self.outputs = np.array([outputs])

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

        The position factor metric (:math:`PF_{r_i \\shortrightarrow r_j}`) is calculated as:

        .. math::

           PF_{r_i \\shortrightarrow r_j} = \\sum_{k=1}^k \\frac{2(P_{k,i} - P_{k,j})}{P_{k,i} + P_{k,j}}

        where:
           :math:`P_{k,i}` is the ranking of factor :math:`k` using :math:`i` trajectories.
           :math:`P_{k,j}` is the ranking of factor :math:`k` using :math:`j` trajectories.

        Parameters
        ----------
        i
            Initial trajectory index from which to start the comparison.
        j
            Final trajectory index against which the comparision is made.
        out_index
            See :meth:`get_metrics`.

        References
        ----------
        Ruano, M. V., Ribes, J., Seco, A., & Ferrer, J. (2012). An improved sampling strategy based on trajectory design
        for application of the Morris method to systems with many input factors. *Environmental Modelling & Software*,
        37, 103–109. https://doi.org/10.1016/j.envsoft.2012.03.008
        """
        mus_i = self[1, :, out_index, :i]
        mus_j = self[1, :, out_index, :j]

        pos_i = np.abs(mus_i.argsort().argsort() - self.k)
        pos_j = np.abs(mus_j.argsort().argsort() - self.k)

        return np.sum(2 * (pos_i - pos_j) / (pos_i + pos_j), axis=1).squeeze()

    @needs_optional_package('matplotlib')
    def plot_sensitivities(self, path: Union[Path, str] = 'sensitivities', out_index: SpecialSlice = 'mean'):
        """ Saves a sensitivity plot.
        The plot is a scatter of :math:`\\mu^*` versus :math:`\\sigma` with dividers between 'important', 'interacting'
        and 'non-influential' categories.

        Parameters
        ----------
        path
            If one plot is being produced (see `out_index`) this is interpreted as the filename with which to save the
            figure. If multiple plots are produced for all the outputs then this is interpreted as a directory into
            which the figures will be saved.
        out_index
            See :meth:`get_metrics`. If :obj:`None`, one plot will be created for each output.

        References
        ----------
        Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban
        water quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*,
        522, 339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056
        """
        self._plotting_core(path, out_index, self._plot_sensitivities_stub)

    @needs_optional_package('matplotlib')
    def plot_rankings(self, path: Union[Path, str] = 'ranking', out_index: SpecialSlice = 'mean'):
        """ Saves the factor rankings as a plot.
        Plots the ordered :math:`\\mu^*` values against their corresponding parameter indices.

        Parameters
        ----------
        See :meth:`plot_sensitivities`.
        """
        self._plotting_core(path, out_index, self._plot_ranking_stub)

    def invert(self):
        """  """

    def _calculate(self,
                   out_index: SpecialSlice,
                   traj_index: Union[int, slice, List[int]]) -> np.ndarray:
        """ Calculates the Estimated Effects metrics (:math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`).

        Parameters
        ----------
        out_index
            See :attr:`metrics`.
        traj_index
            See :attr:`metrics`.

        Returns
        -------
        numpy.ndarray
            :math:`m \\times \\times k \\times h` array of sensitivity metrics where :math:`m` is the metric, :math:`k`
            is the factor and :math:`h` is the output dimensionality.

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
            where = np.where(x_diffs)[0::2]
            x_diffs = np.sum(x_diffs, axis=2)
            y_diffs = self.outputs[traj_index, :-1, out_index] - self.outputs[traj_index, 1:, out_index]

        else:  # Radial style trajectories
            x_diffs = self.trajectories[traj_index, 0] - self.trajectories[traj_index, 1:]
            where = np.where(x_diffs)[0::2]
            x_diffs = np.sum(x_diffs, axis=1)
            y_diffs = self.outputs[traj_index, 0, out_index] - self.outputs[traj_index, 1:, out_index]

        ee = y_diffs / x_diffs[:, :, None]
        ee[where] = ee.copy().ravel().reshape(-1, self.h)

        mu = np.mean(ee, axis=0)
        mu_star = np.mean(np.abs(ee), axis=0)
        sigma = np.std(ee, axis=0, ddof=1)

        return np.array([mu, mu_star, sigma])

    def _plotting_core(self, path: Union[Path, str], out_index: SpecialSlice, plot_stub: Callable[..., None]):
        """ Most plot function require the same looping and gathering of metrics, this is done here and then passed
        to the `plot_stub` method which have the individualized plot commands.
        """
        metrics = np.atleast_3d(self[:, :, out_index])

        path = Path(path)
        is_multi = False
        if metrics.shape[2] > 1:
            path.mkdir(exist_ok=True, parents=True)
            is_multi = True

        for i in range(metrics.shape[2]):
            fig, ax = plt.subplots(figsize=(15, 15))
            fig: plt.Figure
            ax: plt.Axes

            plot_stub(metrics[:, :, i], fig, ax)

            fig.tight_layout()
            fig.savefig(path / f'{i:03}.png' if is_multi else path)
            plt.close(fig)

    def _plot_sensitivities_stub(self, metrics: np.ndarray, fig: plt.Figure, ax: plt.Axes):
        _, ms, sd = metrics

        ax.set_title("Sensitivity classification of all input factors.")
        ax.set_xlabel("$\\mu^*$")
        ax.set_ylabel("$\\sigma$")

        # Influencial / Non-influencial Line
        max_sd = max(sd)
        ax.vlines(self.ct, 0, max_sd, color='red')
        ax.annotate('Non-Influential   ', (self.ct, max_sd), ha='right')

        # Linear / Nonlinear Effect Line
        max_mu = max(ms)
        ax.plot([0, max_mu], [0, max_mu * np.sqrt(self.r) / 2], color='red')
        ax.annotate('   Interacting', (self.ct, max_sd), ha='left')
        ax.annotate('   Important', (self.ct, 0), ha='left')

        # Sensitivities
        ax.scatter(ms, sd, marker='.')
        for j in range(self.dims):
            ax.annotate(j, (ms[j], sd[j]), fontsize=9)

    def _plot_ranking_stub(self, metrics: np.ndarray, fig: plt.Figure, ax: plt.Axes):
        _, ms, _ = metrics

        ax.set_title("Parameter Ranking")
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("$\\mu^*$")

        i_sort = np.argsort(ms)
        ax.bar(i_sort.astype(str), ms[i_sort])
