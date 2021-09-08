import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing.sharedctypes import Value
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .trajectories import make_radial_trajectory_set


class _CallsValidatorCounter:
    """ Automatically counts each evaluation of a function. """

    def __init__(self, func: Callable, eval_counter: Value):
        self.eval_counter = eval_counter
        self.func = func

    def __call__(self, *args, **kwargs) -> Tuple[bool, Any]:
        with self.eval_counter.get_lock():
            self.eval_counter.value += 1
        try:
            y = self.func(*args, **kwargs)
            if np.isfinite(y).all():
                return True, y
        except:
            pass

        return False, None


def unstable_radial_sampling_strategy(func: Callable[[Sequence[float]], Sequence[float]],
                                      r: int,
                                      k: int,
                                      groupings: Optional[np.ndarray] = None,
                                      include_short_range: bool = False,
                                      parallelization: str = 'threads') -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, float]:
    """ Some functions may be unstable and crash when evaluated at a point. To minimize wasted evaluations, generation
    and evaluation can be meshed.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
        A tuple of the trajectories and outputs in a form that can be directly added to ee.addtraj
        Also returns crashed and the evaluation efficiency.
    """
    eval_counter = Value(int, 0, lock=True)
    func = _CallsValidatorCounter(func, eval_counter)

    g = groupings.shape[1]
    l = 2 if include_short_range else 1
    l *= g
    l += 1  # Number of vectors per trajectory

    crashed = []

    final_trajs = []
    final_outs = []

    futs = set()

    executor = ThreadPoolExecutor if parallelization == 'threads' else ProcessPoolExecutor
    executor = executor(os.cpu_count())

    while len(final_trajs) < r:
        gen_trajs = make_radial_trajectory_set(r - len(final_trajs), k, groupings, include_short_range)

        for t in gen_trajs:
            ft = executor.submit(_validate_radial_trajectory, func, t, include_short_range)
            futs.add(ft)

        for ft in as_completed(futs):
            res = ft.result()

            if res['valid'] is True:
                final_trajs.append(res['traj'])
                final_outs.append(res['out'])

            else:
                for t in res['traj']:
                    crashed.append(t)

    final_trajs = np.array(final_trajs)
    final_outs = np.array(final_outs)
    crashed = np.array(crashed)

    return final_trajs, final_outs, crashed, func.eval_count / np.prod(final_outs.shape[:2])


def _validate_radial_trajectory(func: Callable[[Sequence[float]], Sequence[float]],
                                t: np.ndarray,
                                include_short_range: bool) -> Dict[str, Union[bool, np.ndarray]]:
    """  """

    outs = []

    # Validate the base point
    valid, y = func(t[0])
    if not valid:
        return {'valid': False, 'traj': t[0, None]}
    outs.append(y)

    # Validate or move auxiliary points several times
    g = t.shape[1] - 1
    g /= 2 if include_short_range else 1

    aux_pts = []
    for i, x in enumerate(t[1:]):
        moveable_axes = np.argwhere(t[0] - x)
        pt_type = 'aux' if i <= g else 'short'

        valid, x, y = _validate_or_move(pt_type, func, x, t[0], moveable_axes, max_moves=5)

        if not valid:
            return {'valid': False, 'traj': x}

        aux_pts.append(x)
        outs.append(y)


def _validate_or_move(pt_type: str,
                      func: Callable[[Sequence[float]], Sequence[float]],
                      x: np.ndarray,
                      base_pt: np.ndarray,
                      moveable_axes: np.ndarray,
                      max_moves: int = 5) -> Tuple[bool, Optional[np.array], Optional[np.ndarray]]:
    """ Evaluates an auxillary point. Returns if valid, moves it a maximum of `max_moves` times before returning a fail.
    """
    crashed = []
    for i in range(2, max_moves + 2):
        valid, y = func(x)
        if valid:
            return True, x, y

        crashed.append(x)

        if pt_type == 'aux':
            x[moveable_axes] = np.random.random(moveable_axes.size)
        else:
            x = (1 - i / 100) * base_pt + (i / 100) * x

    return False, crashed, None
