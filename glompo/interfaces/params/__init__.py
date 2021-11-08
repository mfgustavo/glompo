from .error_fncs import ReaxFFError, XTBError
from .glompo_opt import GlompoParamsWrapper
from .params_builders import setup_reax_from_classic, setup_reax_from_params, setup_xtb_from_params

__all__ = ("GlompoParamsWrapper",
           "ReaxFFError",
           "XTBError",
           "setup_reax_from_classic",
           "setup_reax_from_params",
           "setup_xtb_from_params",)

try:
    from .optimization import Optimization

    __all__ += ("Optimization",)
except (ImportError, AssertionError):
    pass

# todo make sure params interface docs build correctly.
