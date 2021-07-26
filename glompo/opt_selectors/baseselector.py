""" Abstract class for the construction of selectors used in selecting new optimizers. """

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from ..core.optimizerlogger import BaseLogger
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("BaseSelector",
           "IterSpawnStop")


class BaseSelector(ABC):
    """ Selectors are classes which return an optimizer and its configuration when asked by
        the manager. This selection will then be used to start a new optimizer. The full manager
        is supplied to the selector allowing sophisticated decisions to be designed.
    """

    def __init__(self,
                 *avail_opts: Union[Type[BaseOptimizer],
                                    Tuple[Type[BaseOptimizer], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
                 allow_spawn: Optional[Callable[['GloMPOManager'], bool]] = None):
        """ Parameters
            ----------
            avail_opts
                A set of optimizers available to the minimization, these must be subclasses of the BaseOptimizer
                abstract class in order to be compatible with GloMPO.

                Elements in the set may be:
                1) Subclasses (not instances) of BaseOptimizer.
                2) Tuples of:
                    a) BaseOptimizer subclasses as above;
                    b) Dictionary of kwargs used to initialise a BaseOptimizer instance;
                    c) Dictionary of kwargs for calling the BaseOptimizer().minimize() method.

            allow_spawn
                Optional function sent to the selector which is called with the manager object as argument. If it
                returns False the manager will no longer spawn optimizers.
        """
        self.logger = logging.getLogger('glompo.selector')

        self.avail_opts = []
        for item in avail_opts:
            try:
                if isinstance(item, tuple) and len(item) == 3:
                    opt, init_dict, call_dict = item
                    assert issubclass(opt, BaseOptimizer)

                    if init_dict is None:
                        init_dict = {}
                    assert isinstance(init_dict, dict)

                    if call_dict is None:
                        call_dict = {}
                    assert isinstance(call_dict, dict)

                    if 'workers' not in init_dict:
                        init_dict['workers'] = 1

                    self.avail_opts.append((opt, init_dict, call_dict))

                else:
                    opt = item
                    assert issubclass(opt, BaseOptimizer)
                    self.avail_opts.append((opt, {'workers': 1}, {}))

            except AssertionError as e:
                raise ValueError(f"Cannot parse {item}. Expected:  Union[Type[BaseOptimizer], Tuple[BaseOptimizer, "
                                 f"Dict[str, Any], Dict[str, Any]]] expected.") from e

        if callable(allow_spawn):
            self.allow_spawn = allow_spawn
        else:
            self.allow_spawn = _AlwaysSpawn()

    @abstractmethod
    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: BaseLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None, bool]:
        """ Provided with the manager object and opt_log file of all optimizers, returns a optimizer class
            followed by a dictionary of initialisation keyword arguments and then a dictionary of call kwargs for the
            BaseOptimizer().minimize() method.

            Parameters
            ----------
            manager
                Running manager instance, can be used to read certain counters and state variables.
            log
                Contains the details and iteration history of ever optimizer started thus far.
            slots_available
                Number of threads the manager is allowed to start according to its max_jobs attribute and the number of
                existing threads.
                GloMPO API assumes that the selector will attempt to this parameter and return an optimizer which
                requires less threads than slots_available. If this is not possible then None is returned

            Returns
            -------
            Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]], None, :obj:`False`]
                Optimizer class and configuration parameters. Manager will use this to initialise and start a new
                optimizer.

                :obj:`None` is returned in the case that no available optimizer configurations can satisfy the number of
                worker slots available.

                :obj:`False` is a special return which flags the optimizer to never try and start another optimizer for
                the remainder of the optimization.
        """

    def __contains__(self, item):
        opts = (opt[0] for opt in self.avail_opts)
        return item in opts


class IterSpawnStop:
    """ A useful stub which controls spawning based on the number of function calls used thus far. """

    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def __call__(self, mng: 'GloMPOManager'):
        if mng.f_counter >= self.max_calls:
            return False
        return True


class _AlwaysSpawn:
    def __call__(self, *args, **kwargs):
        return True
