from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from multiprocessing import Event
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
from optsam import GFLS, Hook, Logger, Reporter

from .baseoptimizer import BaseOptimizer, MinimizeResult
from ..common.namedtuples import IterationResult


class GFLSOptimizer(BaseOptimizer):

    @classmethod
    def checkpoint_load(cls: Type['BaseOptimizer'], path: Union[Path, str], opt_id: Optional[int] = None,
                        signal_pipe: Optional[Connection] = None,
                        results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                        backend: str = 'threads') -> 'BaseOptimizer':
        """ Recreates a previous instance of the optimizer suitable to continue a optimization from its previous
            state. Below is a basic implementation which should suit most optimizers, may need to be overwritten.

            Parameters
            ----------
            path: Union[Path, str]
                Path to checkpoint file from which to build from. It must be a file produced by the corresponding
                BaseOptimizer().checkpoint_save method.
            opt_id, signal_pipe, results_queue, pause_flag, workers, backend
                These parameters are the same as the corresponding ones in BaseOptimizer.__init__. These will be
                regenerated and supplied by the manager during reconstruction.
        """

    @property
    def is_restart(self):
        return self._is_restart

    def __init__(self, opt_id: Optional[int] = None, signal_pipe: Optional[Connection] = None,
                 results_queue: Optional[Queue] = None, pause_flag: Optional[Event] = None, workers: int = 1,
                 backend: str = 'threads', logger: bool = False, verbose: bool = False,
                 other_hooks: Optional[Sequence[Hook]] = None, gfls_algo_kwargs: Dict[str, Any] = None):
        """
        Initialisation of the GFLS optimizer wrapper for interface with GloMPO.

        Parameters
        ----------
        logger: bool = False
            If True an optsam Logger Hook will be run along with the optimisation and saved when the class is ended.
        verbose: bool = False
            If True an optsam Reporter Hook will be run along with the optimisation to print progress in realtime.
        other_hooks: Optional[Sequence[Hook]] = None
            Any extra optsam Hook instances which should be manually configured.
        gfls_algo_kwargs: Optional[Dict[str, Any]] = None
            Keyword arguments for the optsam GFLS class. If None, the default arguments are used and a value of 2 is
            used for tr_max.
            Valid settings and defaults:
                tr_max      : 0.5
                xtol        : 1e-3
                ftol        : 1e-7
                constraints : ()
                tr_min      : None
                tr_scale    : 0.9
                noise_scale : 0.1
                pop_size    : None
                diis_mode   : "qrsvd"
                seed        : None
        """
        super().__init__(opt_id, signal_pipe, results_queue, pause_flag, workers, backend)
        self.gfls = None
        self.result = None
        self.stopcond = None

        self.algo_kwargs = gfls_algo_kwargs if gfls_algo_kwargs else {}
        if 'tr_max' not in self.algo_kwargs:
            self.algo_kwargs['tr_max'] = 0.5

        self.hooks = list(other_hooks)
        if logger:
            self.hooks.append(Logger())
        if verbose:
            self.hooks.append(Reporter())

    def minimize(self,
                 function: Callable[[Sequence[float]], float],
                 x0: Sequence[float],
                 bounds: Sequence[Tuple[float, float]],
                 callbacks: Callable = None, **kwargs) -> MinimizeResult:
        """
        Minimizes a function, given an initial list of variable values `x0`, and possibly a list of `bounds` on the
        variable values. The `callbacks` argument allows for specific callbacks such as early stopping.


        """

        # TODO: Resume from checkpoint
        # TODO: Save to checkpoint
        # TODO: Incumbent inject
        # TODO: Define 'function' since it has resids

        def _function_after_ask(ask, function):
            x, aux = ask()
            return x, aux, function(x)

        if not self.is_restart:
            self.logger.info("Setting up fresh GFLS")
            self.algo_kwargs['bounds'] = bounds
            self.gfls = GFLS(np.array(x0, dtype=float), **self.algo_kwargs)
            self.result = MinimizeResult()
            for hook in self.hooks:
                hook.before_start(self.gfls)

        self.logger.debug("Entering optimization loop")

        futures = set()
        while not self.stopcond:
            if self.workers > 1:
                pool_executor = ProcessPoolExecutor if self._backend == 'processes' else ThreadPoolExecutor
                with pool_executor(max_workers=self.workers) as executor:
                    while len(futures) < self.workers:
                        future = executor.submit(_function_after_ask, self.gfls.ask(), function)
                        futures.add(future)
                        self.gfls.state["ncall"] += 1

                    done, not_done = wait(futures, return_when=FIRST_COMPLETED)

                    while len(done) > 0 and not stopcond:
                        future = done.pop()

                        stopcond = self.gfls.tell(*future.result())
                        for hook in self.hooks:
                            new_stopcond = hook.after_tell(self.gfls, stopcond)
                            if new_stopcond is not None:
                                stopcond = new_stopcond

                        if self._results_queue:
                            x, _,
                            result = IterationResult(self._opt_id, self.gfls.itell, 1, )

                    futures = not_done

                    if self._results_queue:
                        for _ in as_completed(submitted.values()):
                            loop += 1
                            self.logger.debug(f"Result {loop}/{len(x)} returned.")
                            self._pause_signal.wait()
                            self.check_messages()
                            if self._stop_called:
                                self.logger.debug("Stop command received during function evaluations.")
                                cancelled = [future.cancel() for future in submitted.values()]
                                self.logger.debug(f"Aborted {sum(cancelled)} calls.")
                                break
                    fx = [future.result() for future in submitted.values() if not future.cancelled()]
            else:
                x, aux = self.gfls.ask()()
                ys = function(x)

            self.gfls.tell(x, aux, ys)

            self.logger.debug("Asking for parameter vectors")
            x, auxs = self.gfls.ask()()
            self.logger.debug("Parameter vectors generated")
            fx = self.parallel_map(function, [x0, *x])

            if len(x) != len(fx):
                self.logger.debug("Unfinished evaluation detected. Breaking out of loop")
                break

            if i == 1:
                self._incumbent = {'x': x0, 'fx': fx[0]}

            self.es.tell(x, fx)
            self.logger.debug("Told solutions")
            self.result.x, self.result.fx = self.es.result[:2]
            if self.result.fx == float('inf'):
                self.logger.warning("CMA iteration found no valid results."
                                    "fx = 'inf' and x = (first vector generated by es.ask())")
                self.result.x = x[0]
            self.logger.debug("Extracted x and fx from result")
            if self.verbose and i % 10 == 0 or i == 1:
                print(f"@ iter = {i} fx={self.result.fx:.2E} sigma={self.es.sigma:.3E}")

            if callbacks and callbacks():
                self.callstop("Callbacks termination.")

            if self._results_queue:
                i_best = np.argmin(fx)
                result = IterationResult(self._opt_id, self.es.countiter, self.popsize, x[i_best], fx[i_best],
                                         bool(self.es.stop()))
                self.push_iter_result(result)
                self.logger.debug("Pushed result to queue")
                self.check_messages()
                self.logger.debug("Checked messages")
                self._pause_signal.wait()
                self.logger.debug("Passed pause test")
            self._customtermination(task_settings)
            self.logger.debug("callbacks called")

            if self._incumbent['fx'] < min(fx) and \
                    self.injection_frequency and i - self.injection_counter > self.injection_frequency:
                self.injection_counter = i
                self.es.inject([self._incumbent['x']], force=self.force_injects)
                print("Incumbent solution injected.")

        self.logger.debug("Exited optimization loop")

        self.result.x, self.result.fx = self.es.result[:2]
        self.result.success = np.isfinite(self.result.fx) and self.result.success
        if self.result.fx == float('inf'):
            self.logger.warning("CMA iteration found no valid results."
                                "fx = 'inf' and x = (first vector generated by es.ask())")
            self.result.x = x[0]

        if self.verbose:
            print(f"Optimization terminated: success = {self.result.success}")
            print(f"Final fx={self.result.fx:.2E}")

        if self._results_queue:
            self.logger.debug("Messaging termination to manager.")
            self.message_manager(0, f"Optimizer convergence {self.es.stop()}")

        if self.es.stop() != "Checkpoint Shutdown":
            if self.keep_files:
                name = 'cma_'
                if self._opt_id:
                    name += f'opt{self._opt_id}_'
                name += 'results.pkl'
                with open(name, 'wb') as file:
                    self.logger.debug("Pickling results")
                    pickle.dump(self.es.result, file)

        return self.result

    def callstop(self, reason: str):
        """ Signal to terminate the minimize loop while still returning a result. """

    def checkpoint_save(self, path: Union[Path, str], force: Optional[Set[str]] = None):
        """ Save current state, suitable for restarting. Path is the location for the file or folder to be constructed.
            Note that only the absolutely critical aspects of the state of the optimizer need to be saved. The manager
            will resupply multiprocessing parameters when the optimizer is reconstructed. Below is a basic
            implementation which should suit most optimizers, may need to be overwritten.

            Parameters
            ----------
            path: Union[Path, str]
                Path to file into which the object will be dumped.
            force: Optional[str]
                Set of variable names which will be forced into the dumped file. Convenient shortcut for overwriting if
                fails for a particular optimizer because a certain variable is filtered out of the data dump.
        """

    def inject(self, x: Sequence[float], fx: float):
        """ If configured to do so, the manager will share the best solution seen by any optimizer with the others
            through this method. The default is to save the iteration into the _incumbent property which the minimize
            algorithm may be able to use in some way.
        """
