def max_spread_subset(*trajectories: np.ndarray, r: int) -> np.ndarray:
    """ From a set of trajectories, returns a subset which maximizes the spread of points explored in the space.

    Parameters
    ----------
    *trajectories
        Array of shape :math:`n \\times k+1 \\times k` where :math:`n` is a large number of :math:`k+1 \\times k`
        trajectories as produced by :func:`make_trajectory`.
    r
        Integer such that :math:`r < n` representing the number of trajectories in the returned subset of
        `trajectories`.

    Returns
    -------
    numpy.ndarray
        Subset of `trajectories` which maximally cover the domain space.

    References
    ----------
    Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer. pp. 115-116. *Wiley*.
    DOI: 10.1002/9780470725184
    """
    # Conforming to literature nomenclature
    M = np.array(trajectories)

    subsets = map(subset_distance, itertools.combinations(M, r))
    i_max = np.argmax([*subsets])
    address = find_index(i_max, len(M), r)

    return M[address]


def find_index(i: int, n: int, r: int) -> Sequence[int]:
    """ Given an index `i`, returns the `r` indices corresponding to the `i`th lexicographic combination of `n`
    elements.
    """
    comb = lambda n, k: math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

    loc = []
    assert i < comb(n, r), "i is larger than the maximum number of possible combinations."

    tot_comb = 0
    top = n
    bot = r
    j = -1
    while len(loc) < r:
        j += 1
        bot -= 1
        top -= 1

        while True:
            new_comb = comb(top, bot)
            if new_comb + tot_comb > i:
                loc.append(j)
                break
            else:
                top -= 1
                j += 1
                tot_comb += new_comb
    return loc


def subset_spread(trajs) -> float:
    """ Returns the measure of spread between a set of trajectories (`trajs`).
    Spread is defined as :math:`D_{1,2,...,m} = \\sqrt{d_{12}^2 + d_{13}^2 + ... d_{ml}^2}` where :math:`d_{ml}` is the
    distance measure between two trajectories (see :func:`traj_distance`).

    References
    ----------
    Campolongo, F., Cariboni, J., & Saltelli, A. (2007). An effective screening design for sensitivity analysis of large
    models. *Environmental Modelling & Software*, 22(10), 1509–1518. https://doi.org/10.1016/j.envsoft.2006.10.004
    """
    return np.sqrt(np.sum(np.array([traj_distance(i, j) for i, j in itertools.combinations(trajs, 2)]) ** 2))
