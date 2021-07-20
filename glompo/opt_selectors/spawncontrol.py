__all__ = ("IterSpawnStop",
           "NOptimizersSpawnStop")


class IterSpawnStop:
    """ Controls spawning based on the number of function calls used thus far.

    Parameters
    ----------
    max_calls
        Maximum number of function calls allowed, after which no more optimizers will be started.
    """

    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def __call__(self, mng: 'GloMPOManager'):
        if mng.f_counter >= self.max_calls:
            return False
        return True


class NOptimizersSpawnStop:
    """ Controls spawning based on the number of optimizers used thus far.

    Parameters
    ----------
    max_opts
        Maximum number of optimizers allowed, after which no more optimizers will be started.
    """

    def __init__(self, max_opts: int):
        self.max_opts = max_opts

    def __call__(self, mng: 'GloMPOManager'):
        if mng.o_counter >= self.max_opts:
            return False
        return True


class _AlwaysSpawn:
    def __call__(self, *args, **kwargs):
        return True
