import re
from copy import copy
from datetime import datetime
from pathlib import Path

__all__ = ("CheckpointingControl",)

from typing import Union


class CheckpointingControl:
    """ Class to setup and control the checkpointing behaviour of the :class:`.GloMPOManager`.
    This class has limited functionality and is mainly a container for various settings. The initialisation arguments
    match the class attributes of the same name.

    Attributes
    ----------
    checkpoint_at_conv : bool
        If :obj:`True` a checkpoint is built when the manager reaches convergence and before it exits.

    checkpoint_at_init : bool
        If :obj:`True` a checkpoint is built at the very start of the optimization. This can make starting duplicate
        jobs easier.

    checkpoint_iter_frequency : float
        Frequency (in number of function evaluations) with which GloMPO will save its state to disk during an
        optimization. Function call based checkpointing not performed if this parameter is not provided.

    checkpoint_time_frequency : float
        Frequency (in seconds) with which GloMPO will save its state to disk during an optimization. Time based
        checkpointing not performed if this parameter is not provided.

    checkpointing_dir : Union[pathlib.Path, str]
        Directory in which checkpoints are saved. Defaults to :code:`'checkpoints'`

        .. important::

           This path is always converted to an absolute path, if a relative path is provided it will be relative to the
           current working directory when this object is created. There is no relation to
           :attr:`.GloMPOManager.working_dir`.

    count : int
        Counter for checkpoint naming patterns which rely on incrementing filenames.

    force_task_save : bool
        Some tasks may pickle successfully but fail to load properly, if this is an issue then setting this
        parameter to :obj:`True` will cause the manager to bypass the pickle task step and immediately attempt the
        :meth:`~.BaseFunction.checkpoint_save` method.

    keep_past : int
        The number of newest checkpoints retained when a new checkpoint is made. Any older ones are deleted.
        Default is -1 which performs no deletion. :code:`keep_past = 0` retains no previous results, only the newly
        constructed checkpoint will exist.

        .. note::

           #. GloMPO will only count the directories in :attr:`checkpointing_dir` and matching the supplied
              :attr:`naming_format`.

           #. Existing checkpoints will only be deleted if the new checkpoint is successfully constructed.

    naming_format : str
        Convention used to name the checkpoints.
        Special keys that can be used:

            ===================   ======================
            Naming Format Key     Checkpoint Name Result
            ===================   ======================
            :code:`'%(date)'`     Current calendar date in YYYYMMDD format
            :code:`'%(year)'`     Year formatted to YYYY
            :code:`'%(yr)'`       Year formatted to YY
            :code:`'%(month)'`    Numerical month formatted to MM
            :code:`'%(day)'`      Calendar day of the month formatted to DD
            :code:`'%(time)'`     Current calendar time formatted to HHMMSS (24-hour style)
            :code:`'%(hour)'`     Hour formatted to HH  (24-hour style)
            :code:`'%(min)'`      Minutes formatted to MM
            :code:`'%(sec)'`      Seconds formatted to SS
            :code:`'%(count)'`    Index count of the number of checkpoints constructed.
                                  Count starts from the largest existing match in :attr:`checkpointing_dir`
                                  or zero otherwise. Formatted to 3 digits.
            ===================   ======================

    raise_checkpoint_fail : bool
        If :obj:`True` a failed checkpoint will cause the manager to end the optimization in error. Note, that GloMPO
        will always write out some data when it terminates. This can be a way of preserving data if the checkpoint
        fails. If :obj:`False` an error in constructing a checkpoint will simply raise a warning and pass.
    """

    def __init__(self,
                 checkpoint_time_frequency: float = float('inf'),
                 checkpoint_iter_frequency: float = float('inf'),
                 checkpoint_at_init: bool = False,
                 checkpoint_at_conv: bool = False,
                 raise_checkpoint_fail: bool = False,
                 force_task_save: bool = False,
                 keep_past: int = -1,
                 naming_format: str = 'glompo_checkpoint_%(date)_%(time)',
                 checkpointing_dir: Union[Path, str] = 'checkpoints'):

        self.checkpoint_time_frequency = checkpoint_time_frequency
        self.checkpoint_iter_frequency = checkpoint_iter_frequency
        self.checkpoint_at_init = checkpoint_at_init
        self.checkpoint_at_conv = checkpoint_at_conv
        self.checkpointing_dir = Path(checkpointing_dir).resolve()
        self.raise_checkpoint_fail = bool(raise_checkpoint_fail)
        self.force_task_save = bool(force_task_save)
        self.keep_past = keep_past
        self.naming_format = naming_format
        self.count = None

        codes = {'%[(]date[)]': 8, '%[(]year[)]': 4, '%[(]yr[)]': 2, '%[(]month[)]': 2, '%[(]day[)]': 2,
                 '%[(]time[)]': 6, '%[(]hour[)]': 2,
                 '%[(]min[)]': 2, '%[(]sec[)]': 2}

        format_re = list(copy(self.naming_format))
        for i, char in enumerate(format_re):
            if any([char == c for c in ('{', '(', '+', '*', '|', '.', '$', ')', '}')]):
                format_re[i] = f'[{char}]'
            if any([char == c for c in ('^', '[', ']')]):
                format_re[i] = rf'\{char}'
        format_re = "".join(format_re)
        for key, digits in codes.items():
            format_re = format_re.replace(key, f'[0-9]{{{digits}}}')
        format_re = format_re.replace('%[(]count[)]', '(?P<index>[0-9]{3})')

        self._naming_format_re = format_re

    def get_name(self) -> str:
        """ Returns a new name for a checkpoint matching the naming format. """

        time = datetime.now()
        name = copy(self.naming_format)
        codes = {'%(date)': '%Y%m%d',
                 '%(year)': '%Y',
                 '%(yr)': '%y',
                 '%(month)': '%m',
                 '%(day)': '%d',
                 '%(time)': '%H%M%S',
                 '%(hour)': '%H',
                 '%(min)': '%M',
                 '%(sec)': '%S'}
        for key, val in codes.items():
            name = name.replace(key, time.strftime(val))

        if self.checkpointing_dir.exists():
            max_index = -1
            matches = [re.match(self._naming_format_re, folder.name) for folder in self.checkpointing_dir.iterdir()]
            for match in matches:
                if match and match.lastgroup == 'index':
                    i = int(match.group('index'))
                    max_index = i if i > max_index else max_index
            self.count = max_index + 1
        else:
            self.count = 0

        name = name.replace('%(count)', f'{self.count:03}')
        self.count += 1

        return name

    def matches_naming_format(self, name: str) -> bool:
        """ Returns :obj:`True` if the provided name matches the pattern in the :attr:`naming_format`. """
        return bool(re.match(self._naming_format_re, name))
