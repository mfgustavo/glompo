User Interventions
==================

GloMPO supports a couple of user interventions during optimization.

Manual Optimizer Termination
----------------------------

Users have manual control of optimizer termination. The user may create stop files in the :attr:`.GloMPOManager.working_dir` which, when detected by the manager, will shutdown the chosen optimizer.

Files must be called ``STOP_x`` where ``x`` is the optimizer ID number. This file name is case-sensitive. Examples include ``STOP_1`` or ``STOP_003``. Note that these files should be empty as they are deleted by the manager once processed.

Manual Checkpointing
--------------------

Users may request a :ref:`checkpoint <Checkpointing>` be made at any time. This is done by creating a file named ``CHKPT`` in :attr:`.GloMPOManager.working_dir`. As above, the file is deleted once detected by the manager so it should be empty.

This can still be used even if :class:`.CheckpointingControl` was not setup in the manager. In this case, its defaults are used and the checkpointing directory will appear in :attr:`.GloMPOManager.working_dir`.
