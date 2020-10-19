""" Useful static functions used throughout GloMPO. """
import inspect
import os
from typing import Optional, Sequence, Tuple, Union, overload

import matplotlib
import numpy as np
import yaml

__all__ = ("nested_string_formatting",
           "is_bounds_valid",
           "distance",
           "glompo_colors",
           "present_memory",
           "LiteralWrapper",
           "FlowList",
           "BoundGroup",
           "literal_presenter",
           "optimizer_selector_presenter",
           "generator_presenter",
           "flow_presenter",
           "bound_group_presenter",
           "unknown_object_presenter",
           "FileNameHandler",
           "CheckpointingError")

""" Sundry Code Stubs """


def nested_string_formatting(nested_str: str) -> str:
    """ Reformat strings produced by the _CombiCore class (used by hunter and checkers) by indenting each level
        depending on its nested level.
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
    """ Checks if provided bounds are valid.
        If True raise_invalid raises an error if the bounds are invalid otherwise a bool is returned.
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
    """ Calculate the straight line distance between two points in Euclidean space. """
    return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2))


@overload
def glompo_colors() -> 'matplotlib.colors.ListedColormap': ...


@overload
def glompo_colors(opt_id: int) -> Tuple[float, float, float, float]: ...


def glompo_colors(opt_id: Optional[int] = None) -> \
        Union['matplotlib.colors.ListedColormap', Tuple[float, float, float, float]]:
    """ Returns a matplotlib Colormap instance containing the custom GloMPO color cycle.
        If opt_id is provided than the specific color at that index is returned instead.
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


def present_memory(bytes_: int, digits: int = 2) -> str:
    """ Accepts an integer number of bytes and returns a string formatted to the most appropriate units. """
    units = 0
    while bytes_ > 1024:
        bytes_ /= 1024
        units += 1

    if units == 0:
        digits = 0

    return f"{bytes_:.{digits}f}{['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'][units]}B"


""" YAML Representers """


class LiteralWrapper(str):
    """ Used by yaml to save some block strings as literals """


class FlowList(list):
    """ Used to wrap lists which should appear in flow style rather than default block style. """


class BoundGroup(list):
    """ Used to better represent the parameter bounds in a human readable but reusable way. """


def literal_presenter(dumper: yaml.Dumper, data: str):
    """ Wrapper around string for better readability in YAML file. """
    return dumper.represent_scalar('tag:yaml.org,2002:str', data.replace(' \n', '\n'), style='|')


def optimizer_selector_presenter(dumper, opt_selector: 'BaseSelector'):
    """ Unique constructor for the optimizer selector. """
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
    """ Unique constructor for the generator. """
    info = {}
    for attr in dir(generator):
        if not attr.startswith('_') and not callable(getattr(generator, attr)) and attr != 'logger' \
                and attr != 'bounds':
            info[attr] = getattr(generator, attr)

    return dumper.represent_mapping('tag:yaml.org,2002:map',
                                    {'Generator': type(generator).__name__,
                                     **info}, flow_style=False)


def flow_presenter(dumper, lst):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', lst, flow_style=True)


def bound_group_presenter(dumper, bound_group):
    grouped = {bound: FlowList([]) for bound in set(bound_group)}
    for i, bound in enumerate(bound_group):
        grouped[bound].append(i)

    return dumper.represent_mapping('tag:yaml.org,2002:map', grouped)


def unknown_object_presenter(dumper, unknown_class: object):
    """ To ensure the YAML file is human readable and can be loaded only with native python types. This constructor
        parses all unknown objects into a dictionary containing their name and instance variables or, if uninitialised,
        just the class name.
    """
    if inspect.isclass(unknown_class):
        return dumper.represent_scalar('tag:yaml.org,2002:str', unknown_class.__name__)

    inst_vars = {}
    for k in dir(unknown_class):
        if not k.startswith('_') and not callable(getattr(unknown_class, k)):
            inst_vars[k] = getattr(unknown_class, k)
    return dumper.represent_mapping('tag:yaml.org,2002:map', {type(unknown_class).__name__: inst_vars})


""" Context Managers """


class FileNameHandler:
    """ Context manager to manage the creation of new files in a different directory from the working one. """

    def __init__(self, name: str):
        """ Decomposes name into a path to a new directory. The final leaf (directory or file) is returned when the
            context manager is created and the working directory is changed to one level up from this final leaf while
            within the context manager. The working directory is returned when exiting the manager.
        """
        self.filename = name
        self.orig_dir = os.getcwd()
        if os.sep in name:
            path, self.filename = name.rsplit(os.sep, 1)
            os.makedirs(path, exist_ok=True)
            os.chdir(path)

    def __enter__(self):
        return self.filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.orig_dir)


""" Custom Errors """


class CheckpointingError(RuntimeError):
    """ Error raised during creation of a checkpoint which would result in an incomplete checkpoint """
