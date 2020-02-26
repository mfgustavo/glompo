

# Native Python imports
import warnings
from typing import *
from time import time
import multiprocessing as mp

# Package imports
from ..common.namedtuples import *
from ..scope.scope import ParallelOptimizerScope
from ..optimizers.baseoptimizer import BaseOptimizer, MinimizeResult
from .gpr import GaussianProcessRegression
from .expkernel import ExpKernel

# Other Python packages
import numpy as np


class GloMPOManager:
    """ Runs given jobs in parallel and tracks their progress using Gaussian Process Regressions.
        Based on these predictions the class will update hyperparameters, kill poor performing jobs and
        intelligently restart others. """

    def __init__(self,
                 task: Callable[[Sequence[float]], float],
                 n_parms: int,
                 optimizers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any], Dict[str, Any]]]],
                 bounds: Sequence[Tuple[float, float]],
                 max_jobs: Optional[int] = None,
                 task_args: Optional[Tuple] = None,
                 task_kwargs: Optional[Dict] = None,
                 convergence_criteria: str = 'sing_conv',
                 x0_criteria: str = 'rand',
                 tmax: Optional[float] = None,
                 omax: Optional[int] = None,
                 fmax: Optional[int] = None,
                 region_stability_check: bool = False,
                 report_statistics: bool = False,
                 history_logging: int = 0,
                 visualisation: bool = False,
                 visualisation_args: Optional[Dict[str, Any]] = None):
        """
        Generates the environment for a globally managed parallel optimization job.

        Parameters
        ----------
        task : Callable[[Sequence[float]], float]
            Function to be minimized. Accepts a 1D sequence of parameter values and returns a single value.
            Note: Must be a standalone function which makes no modifications outside of itself.

        n_parms : int
            The number of parameters to be optimized.

        optimizers : Dict[str, Tuple[Callable, Dict[str, Any]]]
            Dictionary of callable optimization functions (which are children of the BaseOptimizer class) with keywords
            describing their behaviour. Recognized keywords are:
                'default': The default optimizer used if any of the below keywords are not set. This is the only
                    optimizer which *must* be set.
                'early': Optimizers which are more global in their search and best suited for early stage optimization.
                'late': Strong local optimizers which are best suited to refining solutions in regions with known good
                    solutions.
                'noisy': Optimizer which should be used in very noisy areas with very steep gradients or discontinuities
            Values can also optionally be (1, 3) tuples of optimizers and dictionaries of optimizer keywords. The first
            is passed to the optimizer during initialisation and the second during execution.
            For example:
                {'default': CMAOptimizer}: The mp_manager will use only CMA-ES type optimizers in all cases and use default
                    values for them. In cases such as this the optimzer should have no non-default parameters other than
                    func, x0 and bounds.
                {'default': CMAOptimizer, 'late': (CMAOptimizer, {'sigma': 0.1}, None)}: The mp_manager will act as above
                    but in noisy areas CMAOptimizers are initialised with a smaller step size. No special keywords are
                    passed for the minimization itself.

        bounds : Sequence[Tuple[float, float]]
            Sequence of tuples of the form (min, max) limiting the range of each parameter.

        max_jobs : int
            The maximum number of local optimizers run in parallel at one time. Defaults to one less than the number of
            CPUs available to the system with a minimum of 1

        task_args : Optional[Tuple]]
            Optional arguments passed to task with every call.

        task_kwargs : Optional[Dict]
            Optional keyword arguments passed to task with every call.

        convergence_criteria: str
            Criteria used for convergence. Supported arguments:
                'omax': The mp_manager delivers the best answer obtained after initialising omax optimizers. Note that
                    these are not guaranteed to be allowed to terminate themselves i.e. GloMPO may still kill them
                    early.
                'sing_conv': The mp_manager delivers the answer obtained by the first optimizer to be allowed to reach full
                    convergence.
                'n_conv':  The mp_manager delivers the answer obtained by the nth optimizer to be allowed to reach full
                    convergence. Replace n with an integer value.
                    WARNING: It is strongly recommended that this criteria is used with some early termination criteria
                        such as omax, tmax or fmax since later optimizations may not improve on earlier runs and thus
                        may not be allowed to converge, hence this criteria is never reached.

        x0_criteria : str
            Criteria used for selection of the next starting point. Supported arguments:
                'gp': A Guassian process regression is used to estimate areas with low error. Requires many function
                    evaluations (i.e many optimizers started) to begin producing effective suggestions.
                'es': Uses evolutionary strategies of crossover and mutation to generate new starting sets. More
                    reliable and cheaper but does not learn the behaviour of the error function.
                'es+gp': Uses 'es' for the early part of the optimization and switches to 'gp' in the late stage after
                    sufficient information has be acquired.
                'rand': New starting points are selected entirely randomly.

        tmax : Optional[int]
            Maximum number of seconds the optimizer is allowed to run before exiting (gracefully) and delivering the
            best result seen up to this point.
            Note: If tmax is reached GloMPO will not run region_stability_checks.

        omax : Optional[int]
            Maximum number of optimizers (in total) that can be started by the mp_manager.

        fmax : Optional[int]
            Maximum number of function calls that are allowed between all optimizers.

        region_stability_check: bool = False
            If True, local optimizers are started around a candidate solution which has been selected as a final
            solution. This is used to measure its reproducibility. If the check fails, the mp_manager resumes looking for
            another solution.

        report_statistics: bool = False
            If True, the mp_manager reports the statistical significance of the suggested solution.

        history_logging: int = 0
            Indicates the level of logging the user would like:
                0 - No log files are saved.
                1 - The log file of the optimizer from which the final solution was extracted is saved.
                2 - The log file of every started optimizer is saved.
                3 - The log file and screen output of every optimizer is saved.

        visualisation : bool
            If True then a dynamic plot is generated to demonstrate the performance of the optimizers. Further options
            (see visualisation_args) allow this plotting to be recorded and saved as a film.

        visualisation_args : Union[dict, None]
            Optional arguments to parameterize the dynamic plotting feature. See ParallelOptimizationScope.
        """

        print("Hello Here I Am")

        # Save and wrap task
        def task_args_wrapper(func, *args, **kwargs):
            def wrapper(x):
                return func(x, *args, **kwargs)
            return wrapper

        if not callable(task):
            raise TypeError(f"{task} is not callable.")
        self.task = task_args_wrapper(task, task_args, task_kwargs)

        # Save n_parms
        if n_parms > 0 and isinstance(n_parms, int):
            self.n_parms = n_parms
        else:
            raise ValueError(f"Cannot parse n_parms = {n_parms}. Only positive integers are allowed.")

        # Save optimizers
        if 'default' not in optimizers:
            raise ValueError("'default' not found in optimizer dictionary. This value must be set.")
        for key in optimizers:
            if not isinstance(optimizers[key], tuple):
                optimizers[key] = (optimizers[key], None, None)
            if not isinstance(optimizers[key][0], BaseOptimizer):
                raise TypeError(f"{optimizers[key][0]} not an instance of BaseOptimizer.")
        self.optimizers = optimizers

        # Save bounds
        self.bounds = []
        if len(bounds) != n_parms:
            raise ValueError(f"Number of parameters (n_parms) and number of bounds are not equal")
        for bnd in bounds:
            if bnd[0] == bnd[1]:
                raise ValueError(f"Bounds min and max cannot be equal. Rather fix its value and remove it from the "
                                 f"list of parameters. Fixed values can be supplied through task_args or task_kwargs.")
            if bnd[1] < bnd[0]:
                raise ValueError(f"Bound min cannot be larger than max.")
            self.bounds.append(Bound(bnd[0], bnd[1]))

        # Save max_jobs
        if isinstance(max_jobs, int):
            if max_jobs > 0:
                self.max_jobs = max_jobs
            else:
                raise ValueError(f"Cannot parse max_jobs = {max_jobs}. Only positive integers are allowed.")
        else:
            raise TypeError(f"Cannot parse max_jobs = {max_jobs}. Only positive integers are allowed.")

        # Save convergence criteria
        if convergence_criteria in ['omax', 'sing_conv', 'n_conv']:
            self.convergence_criteria = convergence_criteria
        else:
            raise ValueError(f"Cannot parse convergence_criteria = {convergence_criteria}. See docstring for allowed "
                             f"values.")

        # Save x0 criteria
        if x0_criteria in ['gp', 'es', 'es+gp', 'rand']:
            self.x0_criteria = x0_criteria
        else:
            raise ValueError(f"Cannot parse x0_criteria = {x0_criteria}. See docstring for allowed values.")

        # Save max conditions and counters
        self.tmax = np.clip(int(tmax), 1, None) if tmax or tmax == 0 else np.inf
        self.fmax = np.clip(int(fmax), 1, None) if fmax or fmax == 0 else np.inf
        self.omax = np.clip(int(omax), 1, None) if omax or omax == 0 else np.inf

        self.t_counter = None
        self.o_counter = 0
        self.f_counter = 0

        # Save behavioural args
        self.region_stability_check = bool(region_stability_check)
        self.report_statistics = bool(report_statistics)
        self.history_logging = np.clip(int(history_logging), 0, 3)
        if visualisation:
            self.scope = ParallelOptimizerScope(num_streams=max_jobs, **visualisation_args)

        # Setup multiprocessing variables
        self.optimizer_processes = {}
        self.hyperparm_processes = {}
        self.hunting_processes = {}
        self.signal_pipes = {}

        self.mp_manager = mp.Manager()
        self.optimizer_queue = self.mp_manager.Queue()
        self.hyperparm_queue = self.mp_manager.Queue()
        self.hunting_queue = self.mp_manager.Queue()

        # Setup GPRs
        self.gprs = {}

    def start_manager(self) -> MinimizeResult:
        """ Begins the optimization routine.

        Parameters
        ----------
        """
        self.t_counter = time()
        converged = self._check_convergence()
        while self.tmax < time() - self.t_counter and self.o_counter < self.omax and self.f_counter < self.fmax \
                and not converged:
            if mp.active_children() < self.max_jobs:
                self._start_new_job()

        # TODO NB scaling needs to happen in GPR. How? Should this be kernel side or GloMPO side?

    # TODO Selection of new starting points can be driven by a Gaussian Process (Probably unrealistic unless a LOT of
    #  optimizers are started), Genetic Algorithms (Reasonable but still no feeling for the error surface) or random
    #  (very easy to implement and possible same quality as the others given the sparse space).

    def _start_new_job(self):
        self.o_counter += 1
        # TODO we want processes to be daemons. A stuck optimizer somewhere that we havent cleaned up should not stop
        #  us closing the mp_manager.
        pass

    def _kill_job(self, opt_id):
        del self.gprs[opt_id]
        # TODO The signal to end will not be read instantly unless listeners are added to the wrappers
        #  alternatively we can send the signal and if it doesnt co-operate in x time we force it to die.
        #  Optimal solution would be to have both, first solution is 'proper' and the second is an extra layer of
        #  safety.
        warnings.warn(NotImplemented, "Method not implemented.")

    def _start_hunt(self):
        warnings.warn(NotImplemented, "Method not implemented.")

    # TODO Check status/ check health. If we haven't heard from an optimizer in a while we need to make sure the thing
    #  is still running properly. Maybe we need listeners here to detect when a process ends.

    def _optimize_hyperparameters(self):
        warnings.warn(NotImplemented, "Method not implemented.")

    def _explore_basin(self):
        warnings.warn(NotImplemented, "Method not implemented.")

    def _generate_x0(self):
        x0 = []
        if self.x0_criteria == 'gp':
            pass
        elif self.x0_criteria == 'es':
            pass
        elif self.x0_criteria == 'gp+es':
            pass
        else:  # rand
            for i in range(self.n_parms):
                bounds = self.bounds[0]
                x0.append(bounds.min + (bounds.max - bounds.min) * np.random.rand())
        return np.array(x0)

    def _setup_new_optimizer(self) -> OptimizerPackage:
        # TODO add intelligence to pick optimizer?
        selected, init_args, call_args = self.optimizers['default']

        optimizer = selected(**init_args)

        self.gprs[self.o_counter] = GaussianProcessRegression(kernel=ExpKernel(alpha=0.100,
                                                                               beta=5.00),
                                                              dims=1)

        self.signal_pipes[self.o_counter], child_pipe = mp.Pipe()

        return OptimizerPackage(self.o_counter, optimizer, call_args, child_pipe)

    def _check_convergence(self):
        warnings.warn(NotImplemented, "Method not implemented.")
        if self.region_stability_check:
            self._explore_basin()
        return False
