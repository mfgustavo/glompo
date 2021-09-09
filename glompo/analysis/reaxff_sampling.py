import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .trajectories import make_radial_trajectory_set

__all__ = ("unstable_func_radial_sampling_strategy",)


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
        except:
            pass

        return False, None


def unstable_func_radial_sampling_strategy(func: Callable[[Sequence[float]], Sequence[float]],
                                           r: int,
                                           k: int,
                                           groupings: Optional[np.ndarray] = None,
                                           include_short_range: bool = False,
                                           parallelize: bool = True) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """ Unique iterative strategy for unstable functions.
    Some functions may produce non-finite values or errors at specific combinations of the values in input space (e.g.
    :class:`ReaxFFError`). Such points should not be accepted into the sensitivity analysis as they are liable to
    significantly skew the results. This method is an iterative wrapper around :meth:`.make_make_radial_trajectory_set`
    which evaluates trajectory points and only returns validated trajectories for analysis.

    The strategy is as follows:

       #. Generate a selection of trajectories using :meth:`.make_make_radial_trajectory_set`.

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
    Inherited, r k groupings include_include_short_range
        See :meth:`.make_radial_trajectory_set`.
    parallelize
        If :obj:`True` the trajectories will be validated in parallel using threads, otherwise they will be validated
        sequentially.

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
    :meth:`.make_radial_trajectory_set`. However, since the candidate trajectories are still generated in this way, they
    can still be expected to be a close approximation.
    """
    func = _CallsValidatorCounter(func)

    crashed = []
    final_trajs = []
    final_outs = []

    executor = ThreadPoolExecutor(os.cpu_count())

    while len(final_trajs) < r:
        gen_trajs = make_radial_trajectory_set(r - len(final_trajs), k, groupings, include_short_range)
        results = []

        if parallelize:
            futs = set()
            for t in gen_trajs:
                ft = executor.submit(_validate_radial_trajectory, func, t, include_short_range)
                futs.add(ft)

            for ft in as_completed(futs):
                results.append(ft.result())

        else:
            for t in gen_trajs:
                results.append(_validate_radial_trajectory(func, t, include_short_range))

        for res in results:
            if res['valid'] is True:
                final_trajs.append(res['x_valid'])
                final_outs.append(res['y'])
                assert len(final_trajs) <= r

            for x in res['x_crashed']:
                crashed.append(x)

    final_trajs = np.array(final_trajs)
    final_outs = np.atleast_3d(final_outs)
    crashed = np.array(crashed)

    executor.shutdown()

    return final_trajs, final_outs, crashed, np.prod(final_outs.shape[:2]) / func.eval_counter


def _validate_radial_trajectory(func: Callable[[Sequence[float]], Sequence[float]],
                                t: np.ndarray,
                                include_short_range: bool) -> Dict[str, Any]:
    """ Performs the validation procedure for an entire trajectory. """

    valid_pts = []
    valid_outs = []
    crashed = []

    # Validate the base point
    valid, y = func(t[0])
    if not valid:
        return {'valid': False, 'x_crashed': t[0, None]}
    valid_pts.append(t[0])
    valid_outs.append(y)

    # Validate or move auxiliary points several times
    g = t.shape[0] - 1
    g //= 2 if include_short_range else 1

    for i, x in enumerate(t[1:], 1):
        moveable_axes = np.argwhere(t[0] - x).ravel()
        pt_type = 'aux' if i <= g else 'short'

        result = _validate_or_move(pt_type, func, x, t[0], moveable_axes, max_moves=5)

        if not result['valid']:
            return result

        valid_pts.append(result['x_valid'])
        valid_outs.append(result['y'])
        for x_ in result['x_crashed']:
            crashed.append(x_)

    return {'valid': True,
            'x_valid': np.array(valid_pts),
            'y': np.array(valid_outs),
            'x_crashed': crashed}


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
