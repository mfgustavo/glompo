class CheckpointingControl:
    """ Class to setup and control the checkpointing behaviour of GloMPOManagers. """

    def __init__(self,
                 checkpoint_frequency: int,
                 checkpoint_at_conv: bool,
                 keep_past: int = -1,
                 naming_format: str = 'glompo_checkpoint_%(date)_%(time)',
                 checkpointing_dir: str = 'checkpoints'):
        """
        Parameters
        ----------
        checkpoint_frequency: int
            Frequency (in seconds) with which GloMPO will save its state to disk during an optimization. Any such
            directory can be used to initialize a new GloMPOManager and resume an optimization.

        checkpoint_at_conv: bool
            If True a checkpoint is built when the manager reaches convergence and before it exits.

        keep_past: int
            The keep_past newest checkpoints are retained when a new checkpoint is made. Any older ones are deleted.
            Default is -1 which performs no deletion. keep_past = 0 retains no previous results, only the newly
            constructed checkpoint will exist.
            Note, that GloMPO will only count the directories in checkpointing_dir and matching the supplied
            naming_format

        naming_format: str = 'glompo_checkpoint_%(date)_%(time)'
            Convention used to name the checkpoints.
            Special keys that can be used:
                %(date): Current calendar date in YYYYMMDD format
                %(year): Year formatted to YYYY
                %(yr): Year formatted to YY
                %(month): Numerical month formatted to MM
                %(day): Calendar day of the month formatted to DD
                %(time): Current calendar time formatted to HHMMSS
                %(hour): Hour formatted to HH
                %(min): Minutes formatted to MM
                %(sec): Seconds formatted to SS
                %(count): Index count of the number of checkpoints constructed. Count starts from the largest existing
                    folder in the checkpoint_dir or zero otherwise.

        checkpointing_dir: str = 'checkpoints'
            Directory in which checkpoints are saved.
        """

        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_at_conv = checkpoint_at_conv
        self.checkpointing_dir = checkpointing_dir
        self.keep_past = keep_past
        self.naming_format = naming_format

        if '%(count)' in self.naming_format:
            self.count = None
        else:
            self.count = None

    def get_name(self):
        pass
