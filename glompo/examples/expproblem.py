

import numpy as np
from time import sleep


class BaseProblem:
    """Base class for least-squares test problems.

    Two types of noise contributions can be introduced:

    1. "Engine" noise, which is always present. This is normally distributed
       with mean zero and standard deviation sigma_e.
    2. "Measurement" noise, which is only added to the training data. Also
       normally distributed with mean zero and standard deviation sigma_me.

    This class generates synthetic training data for randomly chosen positive
    values of u. The challenge is then to find the parameters p_i.

    """

    # Different types of initial guesses, which should be defined in the subclass.
    GUESSES = {
        "A_easy": None,
        "B_edge": None,
        "C_afar": None,
        "D_lost": None,
        "E_rnd1": None,
        "F_rnd2": None,
    }

    def __init__(self, npar=1, ntrain=50, sigma_me=0.1, sigma_e=0.1, sigma=0.01):
        """Initialize a test case.

        Parameters
        ----------
        npar
            The number of non-linear parameters
        ntrain
            The number of items in the training set
        sigma_me
            Synthetic measurement error added to the training data.
        sigma_e
            Synthetic noise in the non-linear model, to mimic ReaxFF quirks.
        sigma
            The sigma in the cost function.

        Attributes
        ----------
        sigma_e, sigma_me
            See arguments.
        upoints
            The random points on the u-axis for the training set.
        _pars
            The random reference parameters with which the training data were
            generated.
        targets
            The training data (function of upoints and reference parameters).
        sigmas
            The sigmas (tolerances) in the error function.

        """
        self.sigma_me = sigma_me
        self.sigma_e = sigma_e
        self.sigmas = np.full(ntrain, sigma * npar)
        self._pars = self._generate_reference_pars(npar, ntrain, sigma_me)
        self.targets = self._engine_low(self._pars) + np.random.normal(
            0, self.sigma_me * npar, ntrain
        )

    @property
    def npar(self):
        """Number of parameters."""
        return len(self._pars)

    def engine(self, pars, noise=True):
        """Evaluate the (non-linear) "engine".

        Parameters
        ----------
        pars
            One or multiple parameter vectors. ``shape = (npar,) + parshape``.
            The first index should match with the dimensionality of the problem.
            Subsequent indices can be used for vectorization.
        noise
            Controls presence of engine noise.

        Returns
        -------
        fs
            Results of the engine. ``shape = (ntrain,) + parshape``.

        """
        if pars.shape[0] != self.npar:
            raise TypeError("Wrong size of axis 0.")
        result = self._engine_low(pars)
        if noise:
            result += np.random.normal(0, self.sigma_e * self.npar, result.shape)
        return result

    def resids(self, pars, noise=True):
        """Compute normalized residuals

        Parameters
        ----------
        pars
            One or multiple parameter vectors. ``shape = (npar,) + parshape``.
            The first index should match with the dimensionality of the problem.
            Subsequent indices can be used for vectorization.
        noise
            Controls presence of engine noise.

        Returns
        -------
        resids
            Normalized residuals: (f - t)/sigma. ``shape = (ntrain,) + parshape``.

        """
        if pars.shape[0] != self.npar:
            raise TypeError("Wrong size of axis 0.")
        return (self.engine(pars, noise) - self.targets) / self.sigmas

    def _generate_reference_pars(self, npar, ntrain, sigma_me):
        """Generate the training data.

        Subclasses should assign _pars and targets attributes.
        """
        raise NotImplementedError

    def _engine_low(self, pars):
        """Compute engine predictions without noise."""
        raise NotImplementedError


class ExpProblem(BaseProblem):
    r"""A N-dimensional non-linear least-squares problem.

    The model is

    ..math::

        f(\{p_i\}) = \sum_i^npar exp(-p_i u)

    A typical least-squares cost function for this problem has many saddle
    points, making this a challenging test case.

    """

    GUESSES = {
        "A_easy": (lambda npar: np.full(npar, 0.5)),
        "B_edge": (lambda npar: np.ones(npar)),
        "C_afar": (lambda npar: np.full(npar, 3.0)),
        "D_lost": (lambda npar: np.full(npar, 5.0)),
        "E_rnd1": (lambda npar: np.random.uniform(0.1, 1.0, npar)),
        "F_rnd2": (lambda npar: np.random.uniform(0.1, 5.0, npar)),
    }

    def __init__(self, delay: int = 0, npar=1, ntrain=50, sigma_me=0.1, sigma_e=0.1, sigma=0.01):
        self.delay = delay
        super().__init__(npar, ntrain, sigma_me, sigma_e, sigma)

    def _generate_reference_pars(self, npar, ntrain, **kwargs):
        self.upoints = np.random.uniform(0, 10, ntrain)
        _pars = np.random.uniform(0.1, 1.0, npar)
        return _pars

    def _engine_low(self, pars):
        sleep(self.delay)
        return np.exp(-np.einsum("...,i", pars, self.upoints)).sum(axis=0)

    def __call__(self, pars):
        resids = super().resids(pars)
        resids **= 2
        return np.sum(resids)
