from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .trajectories import make_radial_trajectory_set


class _CallsCounted:
    """ Automatically counts each evaluation of a function. """

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def __init__(self, func):
        self._eval_count = 0
        self.func = func

    def __call__(self, *args, **kwargs):
        self._eval_count += 1
        return self.func(*args, *kwargs)


def unstable_radial_sampling_strategy(func: Callable[[Sequence[float]], Sequence[float]],
                                      r: int,
                                      k: int,
                                      groupings: Optional[np.ndarray] = None,
                                      include_short_range: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """ Some functions may be unstable and crash when evaluated at a point. To minimize wasted evaluations, generation
    and evaluation can be meshed.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple of the trajectories and outputs in a form that can be directly added to ee.addtraj
    """
    func = _CallsCounted(func)

    g = groupings.shape[1]
    l = 2 if include_short_range else 1
    l *= g
    l += 1  # Number of vectors per trajectory

    crashed = []

    final_trajs = []
    final_outs = []

    gen_trajs = make_radial_trajectory_set(r - len(final_trajs), k, groupings, include_short_range)

    i = 0  # Indices through gen_traj
    while len(final_trajs) < r:
        if i == gen_trajs.shape[0]:
            gen_trajs = make_radial_trajectory_set(r - len(final_trajs), k, groupings, include_short_range)

        for t in gen_trajs[i:, 0]:
            passed = False
            try:
                y = func(t)
                if np.isfinite(y).all():
                    passed = True
            except:
                pass

            if not passed:
                i += 1
                crashed.append(t)
            else:
                break
