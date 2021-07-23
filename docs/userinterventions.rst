User Interventions
==================

GloMPO supports manual control of optimizer termination. The user may create stop
files in the working directory which, when detected by the manager, will shutdown
the chosen optimizer.

Files must be called `STOP_x` where `x` is the optimizer ID number. This file name
is case-sensitive. Examples include `STOP_1` or `STOP_003`. Note that these files
should be empty as they are deleted by the manager once processed.
