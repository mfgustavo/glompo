import itertools
import os
import sys
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
from time import sleep
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from ..common.helpers import unravel

__all__ = ("make_winding_stair_trajectory",
           "make_radial_trajectory",
           "make_winding_stair_trajectory_set",
           "make_radial_trajectory_set",
           "unstable_func_radial_trajectory_set",)


class _CallsValidatorCounter:
    """ Wrapper class which counts function evaluations and validates them as finite or not.

    Parameters
    ----------
    func
        Function to wrap.

    Attributes
    ----------
    eval_counter
        Number of times the :meth:`!__call__` has been used.
    """

    def __init__(self, func: Callable):
        self._lock = RLock()
        self.eval_counter = 0
        self.func = func

    def __call__(self, *args, **kwargs) -> Tuple[bool, Any]:
        """ Evaluates the function.

        Returns
        -------
        Tuple[bool, Any]
            The first element is :obj:`True` if the function returned a finite numerical value and did not raise an
            error. The second element is the actual function return if the first element is :obj:`True` and :obj:`None`
            otherwise.
        """
        with self._lock:
            self.eval_counter += 1
        try:
            y = self.func(*args, **kwargs)
            if np.isfinite(y).all():
                return True, y
        except Exception as e:
            warnings.warn("Function evaluation raised the following error:\n" + traceback.format_exc(), UserWarning)

        return False, None


def make_winding_stair_trajectory(k: int,
                                  levels: int,
                                  groupings: Optional[np.ndarray] = None,
                                  base_pt: Optional[np.ndarray] = None,
                                  include_short_range: bool = False) -> np.ndarray:
    """ Produces a Manhattan-style trajectory through unit space.
    Generates a set of `k`-dimensional points starting at :math:`\\mathbf{a}` (`base_pt`) and ends at
    :math:`\\mathbf{b}`:

    .. math::

       \\mathbf{S} = \\begin{bmatrix}
                          b_1 & a_2 & ... & a_k \\\\
                          b_1 & b_2 & ... & a_k \\\\
                              &     &\\vdots &  \\\\
                          b_1 & b_2 & ... & b_k \\\\
                     \\end{bmatrix}

    The figure below shows an example of a winding stairs trajectory in three-dimensions. The base point is in red, the
    points in the actual trajectory are in black, and optional short-range points in blue.

    .. image:: ../_static/stairs.png
       :align: center


    Parameters
    ----------
    k
        Number of dimensions in the hypercube the trajectory traverses.
    levels
        Number of levels into which each dimension of the hypercube is discretized.
    groupings
        Optional :math:`k \\times g` matrix of 0s and 1s, mapping parameters to one of :math:`g` groups.
    base_pt
        Optional `k` length vector representing the trajectory starting point. Must be in the lower quadrant of the grid
        such that the addition of the step size in any dimension results in a point which is, itself, in the grid.
        Randomly generated in the grid if not supplied. **Not** included in the final trajectory set.
    include_short_range
        If :obj:`True`, the trajectory will be extended by a further :math:`g` points located at 1% of the distance
        between adjacent points in the trajectory.

    Returns
    -------
    numpy.ndarray
        Sequence of :math:`g + 1` or :math:`2g + 1` vectors in `k` dimensional unit space representing a Manhattan style
        trajectory through it. Where :math:`g \\leq k` is the number of factor groups given by `groupings`.

    Notes
    -----
    The Elementary Effects sensitivity method allows parameter sensitivities to be calculated for groups of parameters,
    not just individual parameters. Trajectories for this type of analysis are produced by providing the `groupings`
    matrix.

    Note that `base_pt` is **not** included in the returned set. This is the opposite behaviour to
    :func:`make_radial_trajectory` but was chosen this way to match literature.

    Including short range points doubles the expense of evaluating the trajectory but provides valuable sensitivity
    information about the smoothness of the surrounding function. Large differences between long-range and short-range
    can indicate an especially pathological function.

    Including short-range analyses can also be critical in residual analyses (see :class:`ResidualAnalysis`).

    References
    ----------
    Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer. pp. 109-128. *Wiley*.
    https://doi.org/10.1002/9780470725184
    """

    # Conforming to literature nomenclature
    p = levels
    d = p / (2 * (p - 1))

    grid = np.linspace(0, 1, p)
    if base_pt:
        grid_set = set(grid[grid < 1 - d])
        assert all([_ in grid_set for _ in set(base_pt)]), "base_pt not in the lower quadrant of the grid."
        x = base_pt
    else:
        x = np.atleast_2d(np.random.choice(grid[grid < 1 - d], (1, k)))

    D = np.diag(np.random.choice([-1, 1], k, True))

    G = np.array(groupings) if groupings is not None else np.identity(k)
    assert G.sum(0).all() and G.sum(1).all() and G.sum() == k, "Every parameter must be in exactly one group."
    g = G.shape[1]

    P = np.identity(g)
    np.random.shuffle(P)

    B = np.tri(g + 1, g, k=-1)
    J = np.ones((g + 1, k))

    path = d / 2 * ((2 * B @ (G @ P).T - J) @ D + J)
    traj = x + path

    assert np.all((0 <= traj) & (traj <= 1))

    if include_short_range:
        traj = np.concatenate([traj, 0.99 * traj[:-1] + 0.01 * traj[1:]])

    return traj


def make_radial_trajectory(k: int,
                           groupings: Optional[np.ndarray] = None,
                           base_pt: Optional[np.ndarray] = None,
                           aux_pt: Optional[np.ndarray] = None,
                           min_dist: float = 0.1,
                           include_short_range: bool = False) -> np.ndarray:
    """ Produces a radial sample set.
    Generates a set of `k`-dimensional points constructed from `base_pt` (:math:`\\mathbf{a}`) and
    `aux_pt` (:math:`\\mathbf{b}`):

    .. math::

       \\mathbf{S} = \\begin{bmatrix}
                          a_1 & a_2 & ... & a_k \\\\
                          b_1 & a_2 & ... & a_k \\\\
                          a_1 & b_2 & ... & a_k \\\\
                              &     &\\vdots &  \\\\
                          a_1 & a_2 & ... & b_k \\\\
                     \\end{bmatrix}

    The figure below shows an example of a radial trajectory in three-dimensions. The base point is in red,
    the auxiliary in green, points in the trajectory are in black, and optional short-range points in blue.

    .. image:: ../_static/radial.png
       :align: center

    Parameters
    ----------
    k
        Number of dimensions in the hypercube.
    groupings
        Optional :math:`k \\times g` matrix of 0s and 1s, mapping parameters to one of :math:`g` groups.
    base_pt
        Optional `k` length vector representing the anchor point to which all other points are compared. Randomly
        generated in unit space if not supplied.
    aux_pt
        Optional `k` length vector representing a point which is never evaluated but whose elements will be substituted
        into the `base_pt` individually in each dimension. Randomly generated in unit space if not supplied.
    min_dist
        The minimum distance between `base_pt` and `aux_pt` along each axis. Used only if `base_pt` and
        `aux_pt` have not been provided.
    include_short_range
        If :obj:`True`, the trajectory will be extended by a further :math:`g` points located at 1% of the distance
        between :math:`\\mathbf{a}` and :math:`\\mathbf{b}` along each axis.

    Returns
    -------
    numpy.ndarray
        Sequence of :math:`g + 1` or :math:`2g + 1` vectors in `k` dimensional unit space representing a collection of
        points sampled around a central point. Where :math:`g \\leq k` is the number of factor groups given by
        `groupings`.

    Notes
    -----
    The Elementary Effects sensitivity method allows parameter sensitivities to be calculated for groups of parameters,
    not just individual parameters. Trajectories for this type of analysis are produced by providing the `groupings`
    matrix.

    Note that `base_pt` is, itself, included in the return. It is the center point to which all other points are
    compared. `aux_pt` is not included in the set and will not be evaluated. This is the opposite behaviour to
    :func:`make_winding_stair_trajectory` but was chosen this way to match literature.

    Including short range points doubles the expense of evaluating the trajectory but provides valuable sensitivity
    information about the smoothness of the surrounding function. Large differences between long-range and short-range
    can indicate an especially pathological function.

    Including short-range analyses can also be critical in residual analyses (see :class:`ResidualAnalysis`).

    References
    ----------
    Campolongo, F., Saltelli, A., & Cariboni, J. (2011). From screening to quantitative sensitivity analysis. A unified
    approach. *Computer Physics Communications*, 182(4), 978–988. https://doi.org/10.1016/J.CPC.2010.12.039
    """
    a = np.array(base_pt) if base_pt is not None else np.random.random(k)

    while aux_pt is None:
        b = np.random.random(k)
        if np.all(np.abs(a - b) > min_dist):
            # Accept a candidate for b only if it is sufficiently far away from a
            break
    else:
        b = aux_pt

    G = np.array(groupings) if groupings is not None else np.identity(k)
    assert G.sum(0).all() and G.sum(1).all() and G.sum() == k, "Every parameter must be in exactly one group."

    where = np.nonzero((np.diag(b) @ G).T)

    long = np.tile(a, (G.shape[1], 1))
    long[where] = b

    if include_short_range:
        short = long.copy()
        short[where] = 0.99 * a + 0.01 * b
    else:
        short = np.empty((0, k))

    A = np.concatenate(([a], long, short), axis=0)

    return A


def make_winding_stair_trajectory_set(r: int,
                                      set_size: int,
                                      k: int,
                                      levels: int,
                                      groupings: Optional[np.ndarray] = None,
                                      include_short_range: bool = False) -> np.ndarray:
    """ Returns a set of well-spaced winding-stair trajectories.
    Samples `set_size` winding-stair trajectories and returns a subset of `r` trajectories which are well-dispersed
    throughout the space.

    Parameters
    ----------
    r
        Number of spaced trajectories to return.
    set_size
        Number of winding-stair trajectories to sample. Higher values increase the chances of having a well dispersed
        set but analyzing its subsets is more expensive.
    Inherited, k levels groupings include_short_range
        See :func:`make_winding_stair_trajectory`.

    Returns
    -------
    numpy.ndarray
        :math:`r \\times g + 1 \\times k` set of trajectories as generated by :func:`make_winding_stair_trajectory`.

    Notes
    -----
    The spread of the trajectories is not globally maximal as would be achieved using the method of
    `Campolongo et al. (2007) <https://doi.org/10.1016/j.envsoft.2006.10.004>`_. However, such a method is horrendously
    expensive due to combinatorial explosion, even at small dimensions. The algorithm used here is much cheaper but
    guaranteed to only be locally maximal in terms of spread. Analysis by
    `Ruano et al (2012) <https://doi.org/10.1016/J.ENVSOFT.2012.03.008>`_ showed the difference to be small.

    This function is distinct from multiple calls to :func:`make_winding_stair_trajectory`, since the trajectories are
    spread in space relative to one another, rather than randomly dispersed.

    References
    ----------
    Ruano, M. V., Ribes, J., Seco, A., & Ferrer, J. (2012). An improved sampling strategy based on trajectory design for
    application of the Morris method to systems with many input factors. *Environmental Modelling & Software*, 37,
    103–109. https://doi.org/10.1016/J.ENVSOFT.2012.03.008.
    """
    M = np.array([make_winding_stair_trajectory(k, levels, groupings, include_short_range=include_short_range)
                  for _ in range(set_size)])

    # Eq (2)
    DM = np.zeros((set_size, set_size))
    for m, t1 in enumerate(M):
        for l, t2 in enumerate(M[m + 1:], m + 1):
            DM[m, l] = DM[l, m] = traj_distance(t1, t2)

    Dr = {}  # Mapping of indices to spread for the r-1 subsets of r trajectories
    for i in range(1, r):
        unused = set(range(set_size))
        loc = []  # Trajectory indicies with the maximum spread for this iteration

        # Eq (3)
        # Starting cluster of i+1 trajectories
        max_col_indices = np.argsort(DM, axis=1)[:, -i:]
        ravel_max_col_indices = np.ravel(max_col_indices)
        row_indices = np.ravel(np.tile(np.arange(set_size), (i, 1)).T)
        Di1 = np.reshape(DM[row_indices, ravel_max_col_indices], (set_size, i))
        Di1 = np.sqrt(np.sum(Di1 ** 2, axis=1))

        # Gather first indices of trajectory
        where_max = np.argmax(Di1)
        loc.append(where_max)
        loc.append(max_col_indices[where_max])
        loc = list(unravel(loc))
        for _ in loc:
            unused.remove(_)

        # Eq 4
        # Expand starting cluster 1-by-1 until r length subsets have been produced
        while len(loc) < r:
            Dik1_max = 0
            k_max = -1
            for k in unused:
                distances = np.array([DM[c] for c in itertools.combinations(loc + [k], 2)])
                Dik1 = np.sqrt(np.sum(distances ** 2))
                if Dik1 > Dik1_max:
                    Dik1_max = Dik1
                    k_max = k
            loc.append(k_max)
            unused.remove(k_max)

        Dr[tuple(sorted(loc))] = Dik1_max

    return M[list(max(Dr, key=Dr.get))]


def make_radial_trajectory_set(r: int,
                               k: int,
                               groupings: Optional[np.ndarray] = None,
                               include_short_range: bool = False) -> np.ndarray:
    """ Returns a set of radial trajectories that are spaced using Latin Hypercube Sampling.

    Parameters
    ----------
    r
        Number of spaced trajectories to generate.
    Inherited, k groupings min_dist include_short_range
        See :func:`make_radial_trajectory`.

    Returns
    -------
    numpy.ndarray
        :math:`r \\times g + 1 \\times k` set of trajectories as generated by :func:`make_radial_trajectory`.

    Notes
    -----
    Literature references generate :math:`\\mathbf{a}` and :math:`\\mathbf{b}` from
    `Sobol Sequences <https://en.wikipedia.org/wiki/Sobol_sequence>`_, however, no Python package (compatible with v3.6)
    is capable to generating Sobol sequences above 40 dimensions. This implementation makes use of Latin-Hypercube
    Sampling instead.

    This function is distinct from multiple calls to :func:`make_radial_trajectory`, since the trajectories are spread
    in space relative to one another, rather than randomly dispersed.
    """
    r2 = 2 * r

    lhs = np.tile(np.arange(r2, dtype=float), (k, 1))
    np.apply_along_axis(np.random.shuffle, 1, lhs)
    lhs += np.random.random(lhs.shape)
    lhs /= r2
    lhs = lhs.T

    part = lambda a: make_radial_trajectory(k, groupings, a[:k], a[k:], include_short_range=include_short_range)
    trajs = np.apply_along_axis(part, 1, np.concatenate([lhs[0::2], lhs[1::2]], 1))

    return trajs


def traj_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    """ Returns a measure of distance between two trajectories.
    Each :math:`(k+1) \\times k` trajectory, `t1` and `t2`, represents a set of points plotting a path through the
    domain. The distance between trajectories is defined as:

    .. math::

        d_{ml} = \\begin{cases}
                        \\sum_{i=1}^{k+1} \\sum_{j=1}^{k+1} \\sqrt{\\sum_{z=1}^k \\left(\\texttt{t1}_{i,j,z} -
                        \\texttt{t2}_{i,j,z}\\right)^2} & \\text{for } m \\neq l \\\\
                        0 & \\text{otherwise}
                 \\end{cases}

    References
    ----------
    Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large
    models. *Environmental Modelling & Software*, 22(10), 1509–1518. https://doi.org/10.1016/j.envsoft.2006.10.004
    """
    paired = np.array([i - j for j in t2 for i in t1])
    return np.sum(np.sqrt(np.sum(paired ** 2, axis=1)))


def unstable_func_radial_trajectory_set(func: Callable[[Sequence[float]], Sequence[float]],
                                        r: int,
                                        k: int,
                                        groupings: Optional[np.ndarray] = None,
                                        include_short_range: bool = False,
                                        parallelize: bool = True,
                                        verbose: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """ Unique iterative strategy for unstable functions.
    Some functions may produce non-finite values or errors at specific combinations of the values in input space (e.g.
    :class:`.ReaxFFError`). Such points should not be accepted into the sensitivity analysis as they are liable to
    significantly skew the results. This method is an iterative wrapper around :meth:`make_radial_trajectory_set`
    which evaluates trajectory points and only returns validated trajectories for analysis.

    The strategy is as follows:

       #. Generate a selection of trajectories using :meth:`make_radial_trajectory_set`.

       #. For each trajectory, evaluate the base point.

       #. If the evaluation of the base point does not produce a finite value, reject the entire trajectory. Otherwise
          proceed.

       #. For each long-range point in the trajectory, evaluate it. If it is not valid, attempt to move it randomly
          (along the same perturbed axis/axes) and repeat the evaluation. Repeat the procedure a maximum of five times
          or until a valid point is found. If a valid point is not found, reject the entire trajectory.

       #. For each short-range point in the trajectory, follow the same procedure as above but instead of moving the
          point randomly, it is incrementally moved along the perturbed axis/axes such that it remains close to the base
          point.

    Parameters
    ----------
    func
        The function for which the trajectory is being generated, and on which the sensitivity analysis will be
        performed. Must accept a :math:`k` length :class:`numpy.ndarray` when called and return an :math:`h` long vector
        of function outputs.
    Inherited, r k groupings include_short_range
        See :meth:`make_radial_trajectory_set`.
    parallelize
        If :obj:`True` the trajectories will be validated in parallel using threads, otherwise they will be validated
        sequentially.
    verbose
        Controls the number of print statements produced by the function. Accepts 0, 1 or 2.
        0 produces no output.
        1 produces a small amount of output suitable for outputs to file.
        2 produces slightly more output suitable for real-time updates via the console.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
        Contains the following:

        #. The validated :math:`r \\times g + 1 \\times k` trajectory.

        #. The :math:`r \\times g + 1 \\times h` array of outputs evaluated for the points in the trajectory.

        #. An :math:`n \\times k` array of vectors which were encountered during the validation which produced errors or
           non-finite values. If analyzed, may be helpful in identifying patterns and causes of problems.

        #. Fraction of function evaluations resulting valid outputs. Gives a measure of the stability of the function.

    Warnings
    --------
    If parallelizing the validation ensure your function is thread-safe!!

    Thread parallelism will have **no** benefit if `func` is entirely Python based and does not internally
    hand-off its work to outside processes. See :ref:`Parallelism` for more details.

    This method does not support multiprocessing based parallelism.

    Notes
    -----
    Due to the possibility of rejecting trajectories and moving points, the returned set of trajectories is no longer
    guaranteed to contain a Latin Hypercube Sample of points within the inputs space as produced by
    :meth:`make_radial_trajectory_set`. However, since the candidate trajectories are still generated in this way, they
    can still be expected to be a close approximation.
    """

    def optional_print(mess, *args, **kwargs):
        if verbose:
            print(mess, *args, **kwargs, flush=True)

    func = _CallsValidatorCounter(func)

    crashed = []
    final_trajs = []
    final_outs = []

    while len(final_trajs) < r:
        optional_print(f"{len(final_trajs)} / {r} valid trajectories found so far.")
        gen_trajs = make_radial_trajectory_set(r - len(final_trajs), k, groupings, include_short_range)
        results = []

        optional_print("---------------------------")
        optional_print(f"{len(gen_trajs)} New trajectories generated.")
        optional_print("Beginning validation...")

        pbar = None
        pfix = {f'{i:02}': 'Queued'.ljust(15, '.') for i, _ in enumerate(gen_trajs)}
        print_lots = verbose == 2
        if verbose:
            pbar = tqdm(total=gen_trajs.shape[0] * (gen_trajs.shape[1] - 1),
                        leave=True,
                        postfix=pfix,
                        file=sys.stdout)

        if parallelize:
            with ThreadPoolExecutor(os.cpu_count()) as executor:
                futs = set()
                for i, t in enumerate(gen_trajs):
                    ft = executor.submit(_validate_radial_trajectory, i, func, t, include_short_range, pbar, print_lots)
                    futs.add(ft)

                if verbose == 1:
                    sleep(0.01)
                    pbar.refresh()

                for i, ft in enumerate(as_completed(futs)):
                    if pbar:
                        pfix = dict((s.split('=') for s in pbar.postfix.split(', ')))
                        pfix[f'{i:02}'] = ft.result()['pbar_message']
                        pbar.set_postfix(pfix, refresh=(i + 1) < len(futs))

                    results.append(ft.result())

        else:
            for i, t in enumerate(gen_trajs):
                res = _validate_radial_trajectory(i, func, t, include_short_range, pbar, print_lots)
                if pbar:
                    pfix = dict((s.split('=') for s in pbar.postfix.split(', ')))
                    pfix[f'{i:02}'] = res['pbar_message']
                    pbar.set_postfix(pfix, refresh=(i + 1) < len(futs))

                results.append(res)

        if pbar:
            pbar.close()

        optional_print("Validation cycle complete...")
        n_accept = 0
        for res in results:
            if res['valid'] is True:
                final_trajs.append(res['x_valid'])
                final_outs.append(res['y'])
                n_accept += 1
                assert len(final_trajs) <= r

            for x in res['x_crashed']:
                crashed.append(x)

        optional_print(f"{n_accept} / {len(results)} accepted as valid trajectories.")

    optional_print("Requested number of trajectories found.")
    final_trajs = np.array(final_trajs)
    final_outs = np.atleast_3d(final_outs)
    crashed = np.array(crashed)

    return final_trajs, final_outs, crashed, np.prod(final_outs.shape[:2]) / func.eval_counter


def _validate_radial_trajectory(traj_id: int,
                                func: Callable[[Sequence[float]], Sequence[float]],
                                t: np.ndarray,
                                include_short_range: bool,
                                pbar: Optional[tqdm],
                                print_lots: bool) -> Dict[str, Any]:
    """ Performs the validation procedure for an entire trajectory. """

    def update_progress(n):
        if pbar:
            if print_lots:
                pbar.update(n)
            else:
                with pbar.get_lock():
                    pbar.n += n

    if pbar:
        pfix = dict((s.split('=') for s in pbar.postfix.split(', ')))
        pfix[f'{traj_id:02}'] = "Building...".ljust(15, '.')
        pbar.set_postfix(pfix, refresh=print_lots)

    valid_pts = []
    valid_outs = []
    crashed = []

    g = t.shape[0] - 1
    g //= 2 if include_short_range else 1

    # Validate the base point
    valid, y = func(t[0])
    if not valid:
        update_progress(t.shape[0] - 1)
        return {'valid': False, 'x_crashed': t[0, None], 'pbar_message': "Base pt fail".ljust(15, '.')}
    valid_pts.append(t[0])
    valid_outs.append(y)

    # Validate or move auxiliary points several times
    for i, x in enumerate(t[1:], 1):
        moveable_axes = np.argwhere(t[0] - x).ravel()
        pt_type = 'aux' if i <= g else 'short'

        result = _validate_or_move(pt_type, func, x, t[0], moveable_axes, max_moves=5)

        if not result['valid']:
            result['pbar_message'] = "Max moves fail".ljust(15, '.')
            update_progress(t.shape[0] - (i if print_lots else 1))
            return result

        if pbar and print_lots:
            pbar.update(1)
        valid_pts.append(result['x_valid'])
        valid_outs.append(result['y'])
        for x_ in result['x_crashed']:
            crashed.append(x_)

    if pbar and not print_lots:
        pbar.n += t.shape[0] - 1

    return {'valid': True,
            'x_valid': np.array(valid_pts),
            'y': np.array(valid_outs),
            'x_crashed': crashed,
            'pbar_message': "Success".ljust(15, '.')}


def _validate_or_move(pt_type: str,
                      func: Callable[[Sequence[float]], Sequence[float]],
                      x: np.ndarray,
                      base_pt: np.ndarray,
                      moveable_axes: np.ndarray,
                      max_moves: int = 5) -> Dict[str, Any]:
    """ Performs the validation procedure on a non-base point in a trajectory.

    Parameters
    ----------
    pt_type
        :code:`aux` or :code:`short` to signify whether the point should be treated as a long-range jump which can be
        arbitrarily, or a short-range one which is only moved incrementally.
    func
        Function which evaluates the trajectory points.
    x
        Point being validated or moved.
    base_pt
        Base point of the trajectory for reference when moving short-range points.
    moveable_axes
        Factors of x which differ from the `base_pt`.
    max_moves
        Maximum number of times the method attempts to move `x` before returning a failed validation.
    """
    crashed = []
    for i in range(2, max_moves + 2):
        valid, y = func(x)
        if valid:
            return {'valid': True,
                    'x_valid': x,
                    'x_crashed': crashed,
                    'y': y}

        crashed.append(x.copy())

        if pt_type == 'aux':
            x[moveable_axes] = np.random.random(moveable_axes.size)
        else:
            x += (x - base_pt) / (i - 1)

    return {'valid': False, 'x_crashed': crashed}
