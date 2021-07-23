from glompo.benchmark_fncs import Michalewicz
from glompo.core.manager import GloMPOManager
from glompo.opt_selectors import CycleSelector
from glompo.optimizers.cmawrapper import CMAOptimizer

if __name__ == '__main__':
    task = Michalewicz(dims=5)

    sigma = (task.bounds[0][1] - task.bounds[0][0]) / 2
    call_args = {'sigma0': sigma}
    init_args = {'workers': 1, 'popsize': 6}
    selector = CycleSelector((CMAOptimizer, init_args, call_args))

    manager = GloMPOManager.new_manager(task=task, bounds=task.bounds, opt_selector=selector)
    result = manager.start_manager()

    print(f"Global min for Michalewicz Function: {task.min_fx:.3E}")
    print("GloMPO minimum found:")
    print(result)
