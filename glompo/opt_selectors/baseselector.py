

""" Abstract class for the construction of selectors used in selecting new optimizers. """


from typing import *
from abc import ABC, abstractmethod

from ..optimizers.baseoptimizer import BaseOptimizer
from ..core.logger import Logger


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
                    b) Dictionary of kwargs used to initialise a BaseOptimizer instance (if there are no arguments
                       use an empty dictionary not None);
                    c) Dictionary of kwargs for calling the BaseOptimizer().minimize() method (if there are no arguments
                       use an empty dictionary not None).
        """
        if not isinstance(avail_opts, list):
            raise TypeError("avail_opts must be a list.")

        self.avail_opts = []
        for item in avail_opts:
            if issubclass(item, BaseOptimizer):
                self.avail_opts.append((item, {}, {}))
            elif isinstance(item, tuple) and issubclass(item[0], BaseOptimizer) and isinstance(item[1], dict) and \
                    isinstance(item[2], dict):
                self.avail_opts.append(item)
            else:
                raise ValueError(f"Cannot parse {item}. Expected:  Union[Type[BaseOptimizer], Tuple[BaseOptimizer, "
                                 f"Dict[str, Any], Dict[str, Any]]] expected.")

    def glompo_log_repr(self):
        """ Returns a representation of this class in dictionary format which is human-readable and can be saved as
            part of the GloMPO manager log yaml file.
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
                         log: Logger) -> Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]]:
        """ Provided with the manager object and log file of all optimizers, returns a optimizer object along with
            initialisation and call kwargs.

            Parameters
            ----------
            manager: GloMPOManager
                Running manager instance, can be used to read certain counters and state variables.
            log: Logger
                Contains the details and iteration history of ever optimizer started thus far.

            Returns
            -------
            optimizer: Type[BaseOptimizer]
                An uninitialised optimizer class.
            init_kwargs: Dict[str, Any]
                Dictionary of kwargs used to initialise the optimizer.
            call_kwargs: Dict[str, Any]
                Dictionary of kwargs passed to the BaseOptimizer().minimize() method when it is called.
        """
