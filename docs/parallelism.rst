.. _Parallelism:

***********
Parallelism
***********

The appropriateness of each depends on the task itself. Using multiprocessing may provide computational
advantages but becomes resource expensive as the task is duplicated between processes, there may also be
I/O collisions if the task relies on external files during its calculation.

If threads are used, make sure the task is thread-safe! Also note that forced terminations are not
possible in this case and hanging optimizers will not be killed. The :attr:`force_terminations_after`
parameter is ignored.

in cases where two levels of parallelism exist (i.e. the optimizers and multiple parallel function
evaluations
therein). Then both levels can be configured to use processes to ensure adequate resource distribution by
launching optimizers non-daemonically. By default the second parallelism level is threaded