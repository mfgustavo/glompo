# Native Python
from typing import *
import warnings
import numpy as np
import os

# AMS
from scm.params.core.parameteroptimization import Optimization
from scm.params.core.lossfunctions import Loss

# Optsam Package
from optsam.fwrap import ResidualsWrapper
from optsam.codec import VectorCodec, BoxTanh
from optsam.opt_gfls import GFLS
from optsam.driver import driver

# This Package
from .baseoptimizer import BaseOptimizer, MinimizeResult


class GFLSOptimizer(BaseOptimizer):

    needscaler = False

    def __init__(
        self,
        opt_id: int,
        tmax: Union[int, None] = None,
        imax: Union[int, None] = None,
        fmax: Union[int, None] = None,
        verbose: int = 30,
        save_logger: Union[str, None] = None,
        gfls_kwargs: Union[dict, None] = None,
    ):
        """ Initialises some configurations of the GFLS optimizer that cannot be passed through ParAMS.

        Parameters
        ----------
        opt_id : int
            Unique ID of the optimizer.
        tmax
            Stopping condition for the wall time in seconds. The optimization will
            stop when the given time has passed after one of the iterations. The
            actual time spent may be a bit longer because an ongoing iteration
            will not be interrupted.
        imax
            Stopping condition for the number of iterations.
        fmax
            Stopping condition for the number of function calls to the wrapper.
            This condition is checked after an iteration has completed. Function
            calls needed for the initialization are not counted.
        verbose
            When zero, no screen output is printed. If non-zero, the integer
            determines the frequency of printing the header of the logger.
        save_logger
            An optional string which if provided saves the output of the logger to the filename given.
        gfls_kwargs
            Arguments passed to the setup of the GFLS class. See opt_gfls.py or documentation.
        """
        super().__init__(opt_id)
        self.tmax: int = tmax
        self.imax: int = imax
        self.fmax: int = fmax
        self.verbose: bool = verbose
        self.save_logger: bool = save_logger
        self.algorithm = GFLS(**gfls_kwargs)

    # noinspection PyMethodOverriding
    def minimize(
        self,
        function: Callable,
        x0: Sequence[float],
        bounds: Sequence[Tuple[float, float]],
        callbacks: Callable = None,
    ) -> MinimizeResult:

        if callbacks:
            warnings.warn(
                "Callbacks are not supported by the GFLS optimizer. Please use options in the initialisation "
                "method to control its behaviour.",
                UserWarning,
            )

        gfls_bounds = []
        for bnd in bounds:
            if bnd[0] == bnd[1]:
                raise ValueError("Min and Max bounds cannot be equal. Rather fix the value and set the variable"
                                 "inactive through the interface.")
            else:
                gfls_bounds.append(BoxTanh(bnd[0], bnd[1]))
        vector_codec = VectorCodec(gfls_bounds)

        for i, x in enumerate(x0):
            if x < bounds[i][0] or x > bounds[i][1]:
                raise ValueError("x0 values outside of bounds.")

        fw = ResidualsWrapper(function.resids, vector_codec.decode)
        logger = driver(
            fw,
            vector_codec.encode(x0),
            self.algorithm,
            self.tmax,
            self.imax,
            self.fmax,
            self.verbose
        )
        if self.save_logger:
            if "/" in self.save_logger:
                path, name = tuple(self.save_logger.rsplit("/", 1))
                os.makedirs(path)
            else:
                name = self.save_logger
            logger.save(name)

        cond = logger.aux["stopcond"]
        success = True if any(cond == k for k in ["xtol", "tr_min"]) else False
        fx = logger.get("func_best", -1)
        history = logger.get_tracks("func")[0]
        index = np.where(history == fx)[0][0]
        x = logger.get("pars", index)

        result = MinimizeResult()
        result.success = success
        result.x = vector_codec.decode(x)
        result.fx = fx

        return result


class ResidualLossFunction(Loss):
    """ Special cost function which returns a numpy array of residual errors. Must be used in conjunction with the GFLS
        optimizer.
    """

    def eval_entry(self, weight, y_ref: float, y_pred: float = None) -> float:
        if y_pred is None:
            ret = y_ref / weight  # Comparator
        else:
            ret = (y_ref - y_pred) / weight  # No comparator
        self.contribution.append(ret)
        return ret

    def eval_final(self) -> np.ndarray:
        self.fx = np.sum(np.array(self.contribution) ** 2)
        return self.fx
