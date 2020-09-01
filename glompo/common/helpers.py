""" Useful static functions used throughout GloMPO. """
import os
from typing import Sequence, Tuple

import numpy as np
import yaml

__all__ = ("LiteralWrapper",
           "FileNameHandler",
           "nested_string_formatting",
           "is_bounds_valid",
           "literal_presenter",
           "distance")


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


def literal_presenter(dumper: yaml.Dumper, data: str):
    """ Wrapper around string for correct presentation in YAML file. """
    return dumper.represent_scalar('tag:yaml.org,2002:str', data.replace(' \n', '\n'), style='|')


def distance(pt1: Sequence[float], pt2: Sequence[float]):
    return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2))


class LiteralWrapper(str):
    """ Used by yaml to save some block strings as literals """


class FileNameHandler:
    def __init__(self, name: str):
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
