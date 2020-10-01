import os
import re
from copy import copy
from datetime import datetime

__all__ = ("CheckpointingControl",)


class CheckpointingControl:
    """ Class to setup and control the checkpointing behaviour of GloMPOManagers. """

    def __init__(self,
                 checkpoint_frequency: int,
                 checkpoint_at_init: bool,
                 checkpoint_at_conv: bool,
                 raise_checkpoint_fail: bool = False,
                 keep_past: int = -1,
                 naming_format: str = 'glompo_checkpoint_%(date)_%(time)',
                 checkpointing_dir: str = 'checkpoints'):
        """
        Parameters
        ----------
        checkpoint_frequency: int
            Frequency (in seconds) with which GloMPO will save its state to disk during an optimization. Any such
            directory can be used to initialize a new GloMPOManager and resume an optimization.

        checkpoint_at_init: bool
            If True a checkpoint is built at the very start of the optimization. This can make starting duplicate jobs
            easier.

        checkpoint_at_conv: bool
            If True a checkpoint is built when the manager reaches convergence and before it exits.

        raise_checkpoint_fail: bool = False
            If True a failed checkpoint will cause the manager to end the optimization in error. Note, that GloMPO will
            always write out some data when it terminates. This can be a way of preserving data if the checkpoint fails.
            If False an error in constructing a checkpoint will simply raise a warning and pass.

        keep_past: int
            The keep_past newest checkpoints are retained when a new checkpoint is made. Any older ones are deleted.
            Default is -1 which performs no deletion. keep_past = 0 retains no previous results, only the newly
            constructed checkpoint will exist.
            Notes:
                1) GloMPO will only count the directories in checkpointing_dir and matching the supplied naming_format
                2) Existing checkpoints will only be deleted if the new checkpoint is successfully constructed.

        naming_format: str = 'glompo_checkpoint_%(date)_%(time)'
            Convention used to name the checkpoints.
            Special keys that can be used:
                %(date): Current calendar date in YYYYMMDD format
                %(year): Year formatted to YYYY
                %(yr): Year formatted to YY
                %(month): Numerical month formatted to MM
                %(day): Calendar day of the month formatted to DD
                %(time): Current calendar time formatted to HHMMSS (24-hour style)
                %(hour): Hour formatted to HH  (24-hour style)
                %(min): Minutes formatted to MM
                %(sec): Seconds formatted to SS
                %(count): Index count of the number of checkpoints constructed. Count starts from the largest existing
                    folder in the checkpoint_dir or zero otherwise. Formatted to 3 digits.

        checkpointing_dir: str = 'checkpoints'
            Directory in which checkpoints are saved. NB: If a relative path is provided it will be interrpretted as
            relative to the working directory provided to the manager! It is NOT relative to the working directory where
            the script is executed.
        """

        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_at_init = checkpoint_at_init
        self.checkpoint_at_conv = checkpoint_at_conv
        self.checkpointing_dir = checkpointing_dir
        self.raise_checkpoint_fail = raise_checkpoint_fail
        self.keep_past = keep_past
        self.naming_format = naming_format

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
        self.naming_format_re = format_re
        self.count = None

    def get_name(self):
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

        if not self.count:
            max_index = -1
            matches = [re.match(self.naming_format_re, folder) for folder in os.listdir(self.checkpointing_dir)]
            for match in matches:
                if match:
                    i = int(match.group('index'))
                    max_index = i if i > max_index else max_index
            self.count = max_index + 1

        name = name.replace('%(count)', f'{self.count:03}')
        self.count += 1

        return name

    def matches_naming_format(self, name: str) -> bool:
        """ Returns True if the provided name matches the pattern in the naming_format """
        return bool(re.match(self.naming_format_re, name))
