import functools
import warnings
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np

from ..common.wrappers import needs_optional_package

try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):
    pass

__all__ = ('EstimatedEffects',)


def needs_trajectories(func):
    """ Wrapper to lock access to class functions before trajectories have been added. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args[0].trajectories.size == 0:
            warnings.warn("Please add at least one trajectory before attempting to access calculations.",
                          UserWarning)
            return
        return func(*args, **kwargs)

    return wrapper


def metric_shield(func):
    """ Checks if a metric can be accessed and in sync with the number of trajectories otherwise recalculates all
    metrics before returning.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args[0]._sigma is None:
            args[0].calculate()
        return func(*args, **kwargs)

    return wrapper


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

    """

    @property
    @metric_shield
    def sigma(self) -> Optional[np.ndarray]:
        """ Standard deviation of Estimated Effects for each factor. """
        return self._sigma

    @property
    @metric_shield
    def mu(self) -> Optional[np.ndarray]:
        """ Mean of Estimated Effects for each factor.
        Signed measure of sensitivity.
        """
        return self._mu

    @property
    @metric_shield
    def mu_star(self) -> Optional[np.ndarray]:
        """ Mean of the absolute values of Estimated Effects for each factor.
        Measures magnitude of sensitivity. Used to determine ordering of factors.
        """
        return self._mu_star

    @property
    def r(self) -> int:
        """ The number of trajectories in the set. """
        return len(self.trajectories)

    @property
    def is_converged(self) -> bool:
        """ Converged if the instance has enough trajectories for the factor ordering to be stable.
        Returns :obj:`True` if the change in :meth:`position_factor` over the last 10 trajectory entries is smaller
        than :attr:`convergence_threshold`.
        """
        if self.r <= 10:
            return False
        return self.position_factor(self.r - 10, self.r) < self.convergence_threshold

    @property
    @needs_trajectories
    def classification(self) -> Dict[str, Set[int]]:
        """ Returns a dictionary with each factor index classfied as :code:`'important'`, :code:`'interacting'` and
        :code:`'non-influential'`. Follows the definitive classification system of
        `Vanrolleghem et al. (2105) <https://doi.org/10.1016/J.JHYDROL.2014.12.056>`_.

        Categories are defined as follows:

        :code:`'important'`:
           These factors have a linear effect on the model output.

        :code:`'interacting'`:
           These factors have a nonlinear effect on the model output.

        :code:`'non-influential'`:
            These factors do not have a significant impact on model output.

        References
        ----------
        Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban
        water quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*,
        522, 339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056
        """
        return {'important': set(
            np.argwhere((self.mu_star > self.ct) & (self.sigma < self.mu_star * np.sqrt(self.r) / 2)).ravel()),
                'interacting': set(
                    np.argwhere((self.mu_star > self.ct) & (self.sigma >= self.mu_star * np.sqrt(self.r) / 2)).ravel()),
                'non-influential': set(np.argwhere(self.mu_star < self.ct).ravel())}

    @property
    @needs_trajectories
    def ranking(self) -> np.ndarray:
        """ Returns factor indices in descending order of their influence on the outputs. """
        return self.mu_star.argsort()[-1::-1] + 1

    def __init__(self, dims: int, convergence_threshold: float,
                 cutoff_threshold: float = 0.1, trajectory_style: str = 'radial'):
        self._sigma: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._mu_star: Optional[np.ndarray] = None

        self.trajectories = np.array([])
        self.outputs = np.array([])
        self.traj_style = trajectory_style

        self.dims: int = dims
        self.convergence_threshold = convergence_threshold
        self.ct = cutoff_threshold

    def add_trajectory(self, trajectory: np.ndarray, outputs: np.ndarray):
        """ Add a trajectory of points and their corresponding model output to the calculation.

        Parameters
        ----------
        trajectory
            A trajectory of points as produced by one of the trajectory generation functions (see :mod:`.trajectories`).
            Should have a shape of :math:`(k+1) \times k` where :math:`k` is the number of factors / dimensions of the
            input.
        outputs
            :math:`k+1` model outputs corresponding to the points in the `trajectory`.

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
        self._sigma = None
        self._mu = None
        self._mu_star = None

        if trajectory.shape != (self.dims + 1, self.dims):
            raise ValueError(f"Cannot parse trajectory with shape {trajectory.shape}, must be ({self.dims + 1}, "
                             f"{self.dims}).")
        if outputs.shape != (self.dims + 1,):
            raise ValueError(f"Cannot parse outputs with shape {outputs.shape}, must be ({self.dims + 1}, )")

        try:
            self.trajectories = np.append(self.trajectories, [trajectory], axis=0)
            self.outputs = np.append(self.outputs, [outputs], axis=0)
        except ValueError:
            self.trajectories = np.array([trajectory])
            self.outputs = np.array([outputs])

    def build_until_convergence(self,
                                func: Callable[[np.ndarray], float],
                                r_max: int):
        """  """
        # TODO Implement
        raise NotImplementedError

    @needs_trajectories
    def position_factor(self, i: int, j: int) -> float:
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

        References
        ----------
        Ruano, M. V., Ribes, J., Seco, A., & Ferrer, J. (2012). An improved sampling strategy based on trajectory design
        for application of the Morris method to systems with many input factors. *Environmental Modelling & Software*,
        37, 103–109. https://doi.org/10.1016/j.envsoft.2012.03.008
        """
        mus_i = self.calculate(i)[1]
        mus_j = self.calculate(j)[1]

        pos_i = np.abs(mus_i.argsort().argsort() - len(mus_i))
        pos_j = np.abs(mus_i.argsort().argsort() - len(mus_i))

        return np.sum(2 * (pos_i - pos_j) / (pos_i + pos_j))

    @needs_optional_package('matplotlib')
    @needs_trajectories
    def plot_sensitivities(self, filename: str = 'sensitivities.png'):
        """ Saves a sensitivity plot.
        The image is saved to `filename`. The plot is a scatter of :math:`\\mu^*` versus :math:`\\sigma` with dividers
        between 'important', 'interacting' and 'non-influential' categories.

        References
        ----------
        Vanrolleghem, P. A., Mannina, G., Cosenza, A., & Neumann, M. B. (2015). Global sensitivity analysis for urban
        water quality modelling: Terminology, convergence and comparison of different methods. *Journal of Hydrology*,
        522, 339–352. https://doi.org/10.1016/J.JHYDROL.2014.12.056
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        fig: plt.Figure
        ax: plt.Axes

        ax.set_title("Sensitivity classification of all input factors.")
        ax.set_xlabel("$\\mu^*$")
        ax.set_ylabel("$\\sigma$")

        # Influencial / Non-influencial Line
        ax.vlines(self.ct, 0, max(self.sigma), color='red')
        ax.annotate('Non-Influential   ', (self.ct, max(self.sigma)), ha='right')

        # Linear / Nonlinear Effect Line
        max_mu = max(self.mu_star)
        ax.plot([0, max_mu], [0, max_mu * np.sqrt(self.r) / 2], color='red')
        ax.annotate('   Interacting', (self.ct, max(self.sigma)), ha='left')
        ax.annotate('   Important', (self.ct, 0), ha='left')

        # Sensitivities
        ax.scatter(self.mu_star, self.sigma, marker='.')
        for i in range(self.dims):
            ax.annotate(i, (self.mu_star[i], self.sigma[i]), fontsize=9)

        fig.tight_layout()
        fig.savefig(filename)

    @needs_optional_package('matplotlib')
    @needs_trajectories
    def plot_rankings(self, filename: str = 'rankings.png'):
        """ Saves the factor rankings as a plot.
        The image is saved to `filename`. Plots the ordered :math:`\\mu^*` values against their corresponding parameter
        indices.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        fig: plt.Figure
        ax: plt.Axes

        ax.set_title("Parameter Ranking")
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("$\\mu^*$")

        i_sort = np.argsort(self.mu_star)
        ax.bar(i_sort.astype(str), self.mu_star[i_sort])

        fig.tight_layout()
        fig.savefig(filename)

    @needs_trajectories
    def calculate(self, i: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Calculates the Estimated Effects metrics (:math:`\\mu`, :math:`\\mu^*` and :math:`\\sigma`).

        Parameters
        ----------
        i
            The number of trajectories to use in the calculation of the metrics. If given, the first `i` trajectories
            will be used, otherwise all trajectories will be used.

        Returns
        -------
        numpy.ndarray
            :math:`\\mu` results for every factor
        numpy.ndarray
            :math:`\\mu^*` results for every factor.
        numpy.ndarray
            :math:`\\sigma` results for every factor.

        Notes
        -----
        If `i` is provided and it is less than all the trajectories already saved in the calculation then the results
        will *not* be saved as the :attr:`mu`, :attr:`mu_star` and :attr:`sigma` attributes of the class.

        If all available trajectories are used, then the results are saved into the above mentioned attributes.

        If a user attempts to access any of the attributes (and the number of trajectories has changed since the last
        call), this method is automatically called for all available trajectories.
        """
        i = i if i is not None else self.r

        if self.traj_style == 'stairs':
            x_diffs = self.trajectories[:i, :-1] - self.trajectories[:i, 1:]
            where = np.where(x_diffs)[0::2]
            x_diffs = np.sum(x_diffs, axis=2)
            y_diffs = self.outputs[:i, :-1] - self.outputs[:i, 1:]

        else:  # Radial style trajectories
            x_diffs = self.trajectories[:i, 0] - self.trajectories[:i, 1:]
            where = np.where(x_diffs)[0::2]
            x_diffs = np.sum(x_diffs, axis=1)
            y_diffs = self.outputs[:i, 0] - self.outputs[:i, 1:]

        ee = y_diffs / x_diffs
        ee[where] = ee.copy().ravel()

        mu = np.mean(ee, axis=0)
        mu_star = np.mean(np.abs(ee), axis=0)
        sigma = np.std(ee, axis=0, ddof=1)

        if i is None or i == self.r:
            self._mu = mu
            self._sigma = sigma
            self._mu_star = mu_star

        return mu, mu_star, sigma
