Checkpointing
=============

This saves its state to disk,
these files can be used by a new :class:`.GloMPOManager` instance to resume. Checkpointing options are provided through
a :class:`.CheckpointingControl` instance.

Checkpointing tries to create an entire image of the GloMPO state, it is the user's
responsibility to ensure that the used optimizers are restartable. Within
`tests/test_optimizers.py` there is the `TestSubclassesGlompoCompatible` class which
can be used to ensure an optimizer is compatible with all of GloMPO's functionality.

The optimization task can sometimes not be reduced to a pickled state depending on
it complexity and interfaces to other codes. GloMPO will first attempt to `pickle`
the object, failing that GloMPO will attempt to call the `checkpoint_save()` function if
the task has such a method. If this also fails the checkpoint is created without the
optimization task. GloMPO can be restarted from an incomplete checkpoint if the
missing components are provided.

Similarly to manual stopping of optimizers, manual checkpoints can also be requested
by created a file named `CHKPT` in the working directory. Note, that this file will
be deleted by the manager when the checkpoint is created.

Checkpointing & Logging
-----------------------

.. caution::
   Please pay close attention to how GloMPO handles log files and loading checkpoints.

The HDF5 log file is not included inside the checkpoints since they can become extremely
large if they are being used to gather lots of data. GloMPO always aims to continue an
optimization by appending data to a matching log file rather than creating a new one. To
do this, the following conditions must be met:
1. A log file called `glompo_log.h5` must be present in the working directory.
2. The log file must contain a key matching the one in the checkpoint.

If a file named `glompo_log.h5` is not present then a warning is issued and GloMPO
will begin logging into a new file of this name.

If the file exists but does not contain a matching key an error will be raised.

It is the user's responsibility to ensure that log files are located and named correctly
in the working directory when loading checkpoints.

.. caution::
   GloMPO will overwrite existing data in if a matching log is found in the working
   directory, but it contains more iteration information that the checkpoint. For
   example, a checkpoint was created after 1 hour of optimization but the manager
   continued until convergence at a later point. If the checkpoint is loaded, it will
   expect a the log file to only have 1 hour worth of data. The only way to load this
   checkpoint (and ensure duplicate iterations are not included in the log) is to remove
   any values in the log which were generated after the checkpoint. To avoid data being
   overwritten, the user can manually copy/rename the log file they wish to retain
   before loading a checkpoint.

.. autoclass:: glompo.core.checkpointing.CheckpointingControl
   :members:
