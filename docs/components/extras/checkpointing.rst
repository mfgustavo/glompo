Checkpointing
=============

This saves its state to disk,
these files can be used by a new :class:`GloMPOManager` instance to resume. Checkpointing options are
provided through a :class:`~glompo.core.checkpointing.CheckpointingControl` instance.
