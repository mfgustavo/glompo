""" Useful static functions and classes used throughout the GloMPO package. """
import inspect
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union, overload

import numpy as np
import tables as tb
import yaml

__all__ = ("SplitOptimizerLogs",
           "nested_string_formatting",
           "is_bounds_valid",
           "distance",
           "glompo_colors",
           "present_memory",
           "rolling_min",
           "unravel",
           "infer_headers",
           "deepsizeof",
           "LiteralWrapper",
           "FlowList",
           "BoundGroup",
           "literal_presenter",
           "optimizer_selector_presenter",
           "generator_presenter",
           "flow_presenter",
           "numpy_dtype_presenter",
           "numpy_array_presenter",
           "bound_group_presenter",
           "unknown_object_presenter",
           "WorkInDirectory",
           "CheckpointingError",
           "StopInterrupt",
           )

""" Sundry Code Stubs """


class SplitOptimizerLogs(logging.Filter):
    """ Splits print statements from child processes and threads to text files.
    If this filter is applied to a :class:`~logging.Handler` on the :code:`'glompo.optimizers'` logger it will
    automatically separate the single :code:`'glompo.optimizers'` logging stream into separate ones for each individual
    optimizer.

    Parameters
    ----------
    filepath
        Directory in which new log files will be located.
    propagate
        If propagate is :obj:`True` then the filter will allow the message to pass through the filter allowing all
        :code:`'glompo.optimizers'` logging to be simultaneously recorded together.
    formatter
        Formatting to be applied in the new logs. If not supplied the :mod:`logging` default is used.

    Examples
    --------
    >>> frmttr = logging.Formatter("%(levelname)s : %(name)s : %(processName)s :: %(message)s")

    Adds individual handlers for each optimizer created. Format for the new handlers is set by :code:`frmttr`
    :code:`propagate=True` sends the message on to :code:`opt_handler` which in this case is :obj:`sys.stdout`.

    >>> opt_filter = SplitOptimizerLogs("diverted_logs", propagate=True, formatter=frmttr)
    >>> opt_handler = logging.StreamHandler(sys.stdout)
    >>> opt_handler.addFilter(opt_filter)
    >>> opt_handler.setFormatter(frmttr)

    Messages of the :code:`'INFO'` level will propogate to :obj:`sys.stdout`.

    >>> opt_handler.setLevel('INFO')
    >>> logging.getLogger("glompo.optimizers").addHandler(opt_handler)

    The level for the handlers made in :class:`SplitOptimizerLogs` is set at the higher level.
    Here :code:`'DEBUG'` level messages will be logged to the files even though :code:`'INFO'` level propagates to
    the console.

    >>> logging.getLogger("glompo.optimizers").setLevel('DEBUG')
    """

    def __init__(self, filepath: Union[Path, str] = "", propagate: bool = False,
                 formatter: Optional[logging.Formatter] = None):
        super().__init__()
        self.opened = set()
        self.filepath = Path(filepath) if filepath else Path.cwd()
        self.filepath.mkdir(parents=True, exist_ok=True)
        self.propagate = int(propagate)
        self.fomatter = formatter

    def filter(self, record: logging.LogRecord) -> int:

        opt_id = int(record.name.replace("glompo.optimizers.opt", ""))

        if opt_id not in self.opened:
            self.opened.add(opt_id)

            if self.fomatter:
                message = self.fomatter.format(record)
            else:
                message = record.getMessage()

            with (self.filepath / f"optimizer_{opt_id}.log").open('w+') as file:
                file.write(f"{message}\n")

            handler = logging.FileHandler(self.filepath / f"optimizer_{opt_id}.log")

            if self.fomatter:
                handler.setFormatter(self.fomatter)

            logger = logging.getLogger(record.name)
            logger.addHandler(handler)

        return self.propagate


def nested_string_formatting(nested_str: str) -> str:
    """ Reformat strings produced by the :class:`!._CombiCore` class.
    :class:`.BaseHunter`\\s and :class:`.BaseChecker`\\s produce strings detailing their last evaluation. This method
    parses and indents each nested level of the string to make it more human readable.

    Parameters
    ----------
    nested_str
        Return produced by :meth:`.BaseHunter.__str__` or :meth:`.BaseHunter.str_with_result`.

    Returns
    -------
    str
        String with added nesting and indenting.

    Examples
    --------
    >>> nested_string_formatting("[TrueHunter() AND\\n"
    ...                          "[[TrueHunter() OR\\n"
    ...                          "[FalseHunter() AND\\n"
    ...                          "[TrueHunter() OR\\n"
    ...                          "FalseHunter()]]]\\n"
    ...                          "OR\\n"
    ...                          "FalseHunter()]]")
    "TrueHunter() AND\\n" \\
    "[\\n" \\
    " [\\n" \\
    "  TrueHunter() OR\\n" \\
    "  [\\n" \\
    "   FalseHunter() AND\\n" \\
    "   [\\n" \\
    "    TrueHunter() OR\\n" \\
    "    FalseHunter()\\n" \\
    "   ]\\n" \\
    "  ]\\n" \\
    " ]\\n" \\
    " OR\\n" \\
    " FalseHunter()\\n" \\
    "]"
    """

    # Strip first and last parenthesis if there
    if nested_str[0] == '[':
        nested_str = nested_str[1:]
    if nested_str[-1] == ']':
        nested_str = nested_str[:-1]

    # Move each level to new line
    nested_str = nested_str.replace('[', '[\n')
    nested_str = nested_str.replace(']', '\n]')

    # Split into lines
    level_count = 0
    lines = nested_str.split('\n')

    # Indent based on number of opening and closing brackets seen.
    for i, line in enumerate(lines):
        if '[' in line:
            lines[i] = f"{' ' * level_count}{line}"
            level_count += 1
            continue
        if ']' in line:
            level_count -= 1
        lines[i] = f"{' ' * level_count}{line}"

    nested_str = "\n".join(lines)

    return nested_str


def is_bounds_valid(bounds: Sequence[Tuple[float, float]], raise_invalid=True) -> bool:
    """ Checks if provided parameter bounds are valid.
    'Valid' is defined as meaning that every lower bound is less than the upper bound and every bound is finite.

    Parameters
    ----------
    bounds
        Sequence of min/max pairs indicating the interval in which the optimizer must search for each parameter.
    raise_invalid
        If :obj:`True` raises an error if the bounds are invalid otherwise a bool is returned.

    Returns
    -------
    bool
        :obj:`True` if the bounds are all valid, :obj:`False` otherwise.

    Raises
    ------
    ValueError
        If `raise_invalid` is :obj:`True` and bounds are invalid.

    Examples
    --------
    >>> is_bounds_valid([(0, 1), (-1, 0)])
    True

    >>> is_bounds_valid([(0, 0), (0, float('inf'))], False)
    False
    """

    for bnd in bounds:
        if bnd[0] >= bnd[1]:
            if raise_invalid:
                raise ValueError("Invalid bounds encountered. Min and max bounds may not be equal nor may they be in"
                                 "the opposite order. ")
            return False

        if not np.all(np.isfinite(bnd)):
            if raise_invalid:
                raise ValueError("Non-finite bounds found.")
            return False

    return True


def distance(pt1: Sequence[float], pt2: Sequence[float]):
    """ Calculate the Euclidean distance between two points.

    Examples
    --------
    >>> distance([0,0,0], [1,1,1])
    1.7320508075688772
    """
    return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2))


@overload
def glompo_colors() -> 'matplotlib.colors.ListedColormap': ...


@overload
def glompo_colors(opt_id: int) -> Tuple[float, float, float, float]: ...


def glompo_colors(opt_id: Optional[int] = None) -> \
        Union['matplotlib.colors.ListedColormap', Tuple[float, float, float, float]]:
    """ Returns a :class:`matplotlib.colors.ListedColormap` containing the custom GloMPO color cycle.
    If `opt_id` is provided than the specific color at that index is returned instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = []
    for cmap in ("tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3", "Dark2"):
        for col in plt.get_cmap(cmap).colors:
            colors.append(col)

    cmap = ListedColormap(colors, "glompo_colormap")
    if opt_id:
        return cmap(opt_id)

    return cmap


def present_memory(bytes_: float, digits: int = 2) -> str:
    """ Accepts an integer number of bytes and returns a string formatted to the most appropriate units.

    Parameters
    ----------
    bytes_
        Number of bytes to write in human readable format
    digits
        Number of decimal places to include in the result

    Returns
    -------
    str
        Converted data quantity and units

    Examples
    --------
    >>> present_memory(123456789, 1)
    '117.7MB'
    """
    units = 0
    while bytes_ > 1024:
        bytes_ /= 1024
        units += 1

    if units == 0:
        digits = 0

    return f"{bytes_:.{digits}f}{['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'][units]}B"


def rolling_min(x: Sequence[float]) -> Sequence[float]:
    """ Returns a vector of shape `x` where each index has been replaced by the smallest number seen thus far when
        reading the list sequentially from left to right.

        Examples
        --------
        >>> rolling_min([3, 4, 5, 6, 2, 3, 4, 1, 2, 3])
        [3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    """
    assert all([isinstance(i, (float, int)) and not math.isnan(i) for i in x]), \
        "Non numerical values found in array, unable to process."

    y = list(x).copy()
    for i, val in enumerate(x[1:], 1):
        y[i] = min(val, y[i - 1])
    return y


def unravel(seq: Union[Any, Sequence[Any]]) -> Iterator[str]:
    """ From a nested sequence of items of any type, return a flatten sequence of items.

    Examples
    --------
    >>> unravel([0, [1], [2, 3, [4, 5, [6], 7], 8, [9]]])
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    try:  # First catch in case seq is not iterable at all
        if isinstance(seq, str):
            yield seq
        else:
            for item in seq:
                if not isinstance(item, str):
                    for nested_item in unravel(item):
                        yield nested_item
                else:
                    yield item
    except TypeError:
        yield seq


def infer_headers(infer_from: Sequence[Any]) -> Dict[str, tb.Col]:
    """ Infers :class:`tables.Col` types based on a function evaluation return.
    Only used if :meth:`.BaseFunction.headers` is not defined. The produced headings are required by
    :class:`.FileLogger`. Default header names will be used: :code:`'result_0', ..., 'result_N'`.

    Parameters
    ----------
    infer_from
        A function evaluation result

    Returns
    -------
    Dict[str, tables.Col]
        Mapping of column names to type

    Examples
    --------
    >>> import tables
    >>> infer_headers((1232.1431423, [213, 345, 675], False, "valid"))
    {'result_0': tables.Float64Col(pos=0),
     'result_1': tables.Int64Col((1, 3), pos=1),
     'result_2': tables.BoolCol(pos=2),
     'result_3': tables.StringCol(280, pos=3)}
    """

    tb_type = {}
    for pos, arg in enumerate(infer_from):
        if isinstance(arg, float):
            tb_type[f'result_{pos}'] = tb.Float64Col(pos=pos)

        elif isinstance(arg, bool):
            tb_type[f'result_{pos}'] = tb.BoolCol(pos=pos)

        elif isinstance(arg, int):
            tb_type[f'result_{pos}'] = tb.Int64Col(pos=pos)

        elif isinstance(arg, str):
            tb_type[f'result_{pos}'] = tb.StringCol(280, pos=pos)

        elif any((isinstance(arg, t) for t in (tuple, list, np.ndarray))):
            arr = np.array(list(arg))  # Needed to properly pack nested arrays and save big memory
            tb_type[f'result_{pos}'] = tb.Col.from_dtype(np.dtype((arr.dtype, arr.shape)), pos=pos)

        elif isinstance(arg, type(None)):
            tb_type[f'result_{pos}'] = tb.Float64Col(pos=pos)

        elif isinstance(arg, complex):
            tb_type[f'result_{pos}'] = tb.ComplexCol(itemsize=16, pos=pos)

        else:
            raise TypeError(f"Cannot resolve type {type(arg)}.")

    return tb_type


def deepsizeof(obj) -> int:
    """ Recursively determines the byte size of an object.
    Any initialised objects (not including Python primitives) must correct implement :meth:`!__sizeof__` for this method
    to work correctly.
    """
    size = sys.getsizeof(obj)
    try:
        for i in obj:
            if isinstance(i, dict):
                size += sys.getsizeof({}) + sum(map(deepsizeof, i.keys())) + sum(map(deepsizeof, i.values()))
            elif isinstance(i, (list, tuple, set, frozenset)):
                size += deepsizeof(i)
    except TypeError:
        pass
    finally:
        return size


""" YAML Representers """


class LiteralWrapper(str):
    """ Used by YAML to save some block strings as literals """


class FlowList(list):
    """ Used to wrap lists which should appear in YAML flow style rather than default block style. """


class BoundGroup(list):
    """ Used to better represent the parameter bounds in a human readable but reusable way in YAML. """


def literal_presenter(dumper: yaml.Dumper, data: str):
    """ Wrapper around string for better readability in YAML file. """
    return dumper.represent_scalar('tag:yaml.org,2002:str', data.replace(' \n', '\n'), style='|')


def optimizer_selector_presenter(dumper, opt_selector: 'BaseSelector'):
    """ Unique YAML constructor for the :class:`.BaseSelector`. """
    opts = {}
    for i, opt in enumerate(opt_selector.avail_opts):
        opts[i] = dict(zip(('type', 'init_kwargs', 'call_kwargs'), opt))

    if isinstance(opt_selector.allow_spawn, object):
        allow_spawn = opt_selector.allow_spawn
    else:
        allow_spawn = opt_selector.allow_spawn.__name__

    return dumper.represent_mapping('tag:yaml.org,2002:map',
                                    {'Selector': type(opt_selector).__name__,
                                     'Allow Spawn': allow_spawn,
                                     'Available Optimizers': opts,
                                     }, flow_style=False)


def generator_presenter(dumper, generator: 'BaseGenerator'):
    """ Unique YAML constructor for the :class:`.BaseGenerator`. """
    info = {}
    for attr in dir(generator):
        if not attr.startswith('_') and not callable(getattr(generator, attr)) and attr != 'logger' \
                and attr != 'bounds':
            info[attr] = getattr(generator, attr)

    return dumper.represent_mapping('tag:yaml.org,2002:map',
                                    {'Generator': type(generator).__name__,
                                     **info}, flow_style=False)


def flow_presenter(dumper, lst):
    """ YAML Presenter for a FlowList style list. """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', lst, flow_style=True)


def numpy_dtype_presenter(dumper, numpy_type):
    """ Unique YAML constructor for :class:`numpy.dtype`. """
    value = numpy_type.item()
    try:
        return getattr(dumper, f'represent_{type(value).__name__}')(value)
    except AttributeError:
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))


def numpy_array_presenter(dumper, numpy_arr):
    """ Unique YAML constructor for :class:`numpy.ndarray`. """
    value = numpy_arr.tolist()
    try:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True)
    except TypeError:
        return dumper.represent_scalar('tag:yaml.org,2002:str', str(value))


def bound_group_presenter(dumper, bound_group):
    """ Unique YAML constructor for :class:`.Bound`. """
    grouped = {f"({bound.min}, {bound.max})": FlowList([]) for bound in set(bound_group)}
    for i, bound in enumerate(bound_group):
        grouped[f"({bound.min}, {bound.max})"].append(i)

    return dumper.represent_mapping('tag:yaml.org,2002:map', grouped)


def unknown_object_presenter(dumper, unknown_class: object):
    """ Parses all remaining classes into strings and python primitives for YAML files.
    To ensure the YAML file is human readable and can be loaded only with native python types. This constructor parses
    all unknown objects into a dictionary containing their name and instance variables or, if uninitialised, just the
    class name.
    """
    if inspect.isclass(unknown_class):
        return dumper.represent_scalar('tag:yaml.org,2002:str', unknown_class.__name__)

    inst_vars = {}
    for k in dir(unknown_class):
        if not k.startswith('_') and not callable(getattr(unknown_class, k)):
            inst_vars[k] = getattr(unknown_class, k)

    if inst_vars:
        return dumper.represent_mapping('tag:yaml.org,2002:map', {type(unknown_class).__name__: inst_vars})
    return dumper.represent_mapping('tag:yaml.org,2002:map', {type(unknown_class).__name__: None})


""" Context Managers """


class WorkInDirectory:
    """ Context manager to manage the creation of new files in a different directory from the working one.

    Parameters
    ----------
    path
        A directory to which the working directory will be changed on entering the context manager. If the directory
        does not exist, it will be created. The working directory is changed back on exiting the context manager.
    """

    def __init__(self, path: Union[Path, str]):
        path = Path(path).resolve()
        self.orig_dir = Path.cwd()
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.orig_dir)


""" Custom Errors """


class CheckpointingError(RuntimeError):
    """ Error raised during creation of a checkpoint which would result in an incomplete checkpoint. """


class StopInterrupt(Exception):
    """ Raised if a file called STOP_ALL is found in the working directory. """
