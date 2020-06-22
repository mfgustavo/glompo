

""" Abstract class for the construction of selectors used in selecting new optimizers. """


import logging
from typing import *
from abc import ABC, abstractmethod

from ..optimizers.baseoptimizer import BaseOptimizer
from ..core.optimizerlogger import OptimizerLogger


__all__ = ("BaseSelector",)


class BaseSelector(ABC):

    def __init__(self,
                 avail_opts: List[Union[Type[BaseOptimizer],
                                        Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]]]):
        """ Parameters
            ----------
            avail_opts: Set[Union[Type[BaseOptimizer], Tuple[BaseOptimizer, Dict[str, Any], Dict[str, Any]]]]
                A set of optimizers available to the minimization, these must be subclasses of the BaseOptimizer
                abstract class in order to be compatible with GloMPO.

                Elements in the set may be:
                1) Subclasses (not instances) of BaseOptimizer.
                2) Tuples of:
                    a) BaseOptimizer subclasses as above;
                    b) Dictionary of kwargs used to initialise a BaseOptimizer instance;
                    c) Dictionary of kwargs for calling the BaseOptimizer().minimize() method.
        """
        self.logger = logging.getLogger('glompo.selector')
        if not isinstance(avail_opts, list):
            raise TypeError("avail_opts must be a list.")

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

            except AssertionError:
                raise ValueError(f"Cannot parse {item}. Expected:  Union[Type[BaseOptimizer], Tuple[BaseOptimizer, "
                                 f"Dict[str, Any], Dict[str, Any]]] expected.")

    def glompo_log_repr(self):
        """ Returns a representation of this class in dictionary format which is human-readable and can be saved as
            part of the GloMPO manager opt_log yaml file.
        """

        dict_form = {}
        for i, item in enumerate(self.avail_opts):
            dict_form[i] = {'type': item[0].__name__,
                            'init_kwargs': item[1],
                            'call_kwargs': item[2]}
        return {'Selector': type(self).__name__,
                'Available Optimizers': dict_form}

    @abstractmethod
    def select_optimizer(self,
                         manager: 'GloMPOManager',
                         log: OptimizerLogger,
                         slots_available: int) -> Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]],
                                                        None]:
        """ Provided with the manager object and opt_log file of all optimizers, returns a optimizer class
            followed by a dictionary of initialisation keyword arguments and then a dictionary of call kwargs for the
            BaseOptimizer().minimize() method.

            Parameters
            ----------
            manager: GloMPOManager
                Running manager instance, can be used to read certain counters and state variables.
            log: OptimizerLogger
                Contains the details and iteration history of ever optimizer started thus far.
            slots_available: int
                Number of threads the manager is allowed to start according to its max_jobs attribute and the number of
                existing threads.
                GloMPO API assumes that the selector will attempt to this parameter and return an optimizer which
                requires less threads than slots_available. If this is not possible then None is returned
        """

    def __contains__(self, item):
        opts = (opt[0] for opt in self.avail_opts)
        return item in opts
