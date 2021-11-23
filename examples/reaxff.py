import logging
import sys

from glompo.convergence import MaxSeconds
from glompo.core.checkpointing import CheckpointingControl
from glompo.core.manager import GloMPOManager
from glompo.generators import PerturbationGenerator
from glompo.hunters import EvaluationsUnmoving
from glompo.opt_selectors import CycleSelector

try:
    from glompo.optimizers.cmawrapper import CMAOptimizer
except ModuleNotFoundError:
    raise ModuleNotFoundError("To run this example the cma package is required.")

try:
    from glompo.interfaces.params import ReaxFFError
except ModuleNotFoundError:
    raise ModuleNotFoundError("To run this example the scm package (part of the Amsterdam Modelling Suite [scm.com]) "
                              "is required.")

if __name__ == '__main__':
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s :: %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger('glompo.manager')
    logger.addHandler(handler)
    logger.setLevel('INFO')

    cost_function = ReaxFFError.from_classic_files('../tests/_test_inputs')
    print("Reparameterizing ReaxFF Force Field for Disulfide Training Set")
    print("--------------------------------------------------------------")
    print("Active Parameters:", cost_function.n_parms, "/", cost_function.n_all_parms)
    print("Initial Error Value:",
          cost_function(cost_function.convert_parms_real2scaled(cost_function.par_eng.active.x)))

    checker = MaxSeconds(session_max=24 * 60 * 60)

    sigma = 0.25
    call_args = {'sigma0': sigma}
    init_args = {'workers': 4, 'popsize': 12}  # Change for better balancing on your available computational resources
    selector = CycleSelector((CMAOptimizer, init_args, call_args))

    max_jobs = 24  # Change for better balancing on your available computational resources

    hunters = EvaluationsUnmoving(25000, 0.01)  # Kill optimizers which are incorrectly focusing

    generator = PerturbationGenerator(x0=cost_function.convert_parms_real2scaled(cost_function.par_eng.active.x),
                                      bounds=cost_function.bounds,
                                      scale=[0.25] * cost_function.n_parms)

    backend = 'threads'  # DO NOT CHANGE

    checkpointing = CheckpointingControl(checkpoint_at_conv=True,
                                         naming_format='customized_completed_%(date)_%(time).tar.gz',
                                         checkpointing_dir="customized_example_outputs")

    manager = GloMPOManager.new_manager(task=cost_function,
                                        bounds=cost_function.bounds,
                                        opt_selector=selector,
                                        working_dir="reaxff_example_outputs",  # Dir in which files are saved
                                        overwrite_existing=True,  # Replaces existing files in working directory
                                        max_jobs=max_jobs,
                                        backend=backend,
                                        convergence_checker=checker,
                                        x0_generator=generator,
                                        killing_conditions=hunters,
                                        hunt_frequency=500,  # Function evaluations between hunts
                                        status_frequency=600,  # Interval in seconds in which a status message is logged
                                        checkpoint_control=checkpointing,  # Saves state for restart
                                        summary_files=3,  # Controls the level of output produced
                                        force_terminations_after=-1,
                                        split_printstreams=True)  # Autosend print statements from opts to files

    result = manager.start_manager()

    print("Final Error Value:", result.fx)
