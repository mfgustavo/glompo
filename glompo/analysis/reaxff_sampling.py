import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .trajectories import make_radial_trajectory_set

__all__ = ("unstable_radial_sampling_strategy",)


class _CallsValidatorCounter:
    """ Automatically counts each evaluation of a function. """

    def __init__(self, func: Callable):
        self._lock = RLock()
        self.eval_counter = 0
        self.func = func

    def __call__(self, *args, **kwargs) -> Tuple[bool, Any]:
        with self._lock:
            self.eval_counter += 1
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
                                      include_short_range: bool = False) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """ Some functions may be unstable and crash when evaluated at a point. To minimize wasted evaluations, generation
    and evaluation can be meshed.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
        A tuple of the trajectories and outputs in a form that can be directly added to ee.addtraj
        Also returns crashed and the evaluation efficiency.
    """
    func = _CallsValidatorCounter(func)

    crashed = []
    final_trajs = []
    final_outs = []

    executor = ThreadPoolExecutor(os.cpu_count())

    while len(final_trajs) < r:
        gen_trajs = make_radial_trajectory_set(r - len(final_trajs), k, groupings, include_short_range)

        futs = set()
        for t in gen_trajs:
            ft = executor.submit(_validate_radial_trajectory, func, t.copy(), include_short_range)
            futs.add(ft)

        for ft in as_completed(futs):
            res = ft.result()

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
    """  """

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
    """ Evaluates an auxillary point. Returns if valid, moves it a maximum of `max_moves` times before returning a fail.
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
