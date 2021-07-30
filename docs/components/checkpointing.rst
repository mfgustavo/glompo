Checkpointing
=============

It is sometimes the case that an optimization proceeds for a longer time than is available on computational
infrastructure. For example, a shared high performance computing center with job limits of 72 hours. To address this,
GloMPO incorporates checkpointing functionality. This constructs a snapshot of a managed optimization at some point in
time and persists it to disk. This file can be loaded by a new :class:`.GloMPOManager` instance at a later time to
resume the optimization.

Checkpointing options are provided to the :class:`.GloMPOManager` through a :class:`.CheckpointingControl` instance.

Checkpointing tries to create an entire image of the GloMPO state, it is the user's
responsibility to ensure that the used optimizers are restartable.

.. tip::

   Within ``tests/test_optimizers.py`` there is the :class:`!TestSubclassesGlompoCompatible` class which can be used
   to ensure an optimizer is compatible with all of GloMPO's functionality.

The optimization task can sometimes not be reduced to a pickled state depending on it complexity and interfaces to other
codes. GloMPO will first attempt to :mod:`pickle` the object, failing that GloMPO will attempt to call
:meth:`~.BaseFunction.checkpoint_save`. If this also fails, the checkpoint is created
without the optimization task. GloMPO can be restarted from an incomplete checkpoint if the missing components are
provided.

Similarly to manual stopping of optimizers (see :ref:`User Interventions`), manual checkpoints can also be requested by
creating a file named ``CHKPT`` in the working directory. Note, that this file will be deleted by the manager when the
checkpoint is created.

Checkpointing & Log Files
-------------------------

.. caution::
   Please pay close attention to how GloMPO handles log files and loading checkpoints.

The HDF5 log file is not included inside the checkpoints since they can become extremely large if they are being used to
gather lots of data. GloMPO always aims to continue an optimization by appending data to a matching log file rather than
creating a new one. To do this, the following conditions must be met:

   #. A log file called `glompo_log.h5` must be present in the working directory.

   #. The log file must contain a key matching the one in the checkpoint.

If a file named `glompo_log.h5` is not present then a warning is issued and GloMPO will begin logging into a new file of
this name.

If the file exists but does not contain a matching key an error will be raised.

It is the user's responsibility to ensure that log files are located and named correctly in the working directory
when loading checkpoints.

.. caution::

   GloMPO will overwrite existing data in if a matching log is found in the working directory, but it contains more
   iteration information that the checkpoint.

For example, a checkpoint was created at the 1000th function evaluation of an optimization, but the manager continued
until convergence after 1398 function evaluations. If the checkpoint is loaded, it will expect a the log file to only
have 1000 function evaluations.

The only way to load this checkpoint (and ensure duplicate iterations are not included in the log) is to remove any
values in the log which were generated after the checkpoint. To avoid data being overwritten, the user can manually
copy/rename the log file they wish to retain before loading a checkpoint.

Checkpointing & Visualisation
-----------------------------

If you are visualizing the optimization (see :ref:`GloMPO Scope`), it is unfortunately not possible to continue saving
a movie through a checkpoint. One *can* continue the visualisation through the checkpoint but the recording will go into
a new file. More specifically, it will go into a file with the same name as was configured before the checkpoint, this
means that the pre-checkpoint movie file is likely to be overwritten. Thus, to save the first movie file, the user
should rename it before continuing an optimization from a checkpoint, or change
:attr:`.GloMPOManager.visualisation_args` when loading the checkpoint. It is possible to stitch together several movies
into a single file at a later stage using `ffmpeg <ffmpeg.org>`_.

Checkpointing Control Settings
------------------------------

.. autoclass:: glompo.core.checkpointing.CheckpointingControl
   :members:
