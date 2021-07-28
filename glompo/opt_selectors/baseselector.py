""" Abstract class for the construction of selectors used in selecting new optimizers. """

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from .spawncontrol import _AlwaysSpawn
from ..core.optimizerlogger import BaseLogger
from ..optimizers.baseoptimizer import BaseOptimizer

__all__ = ("BaseSelector",)


class BaseSelector(ABC):
    """ Base selector from which all selectors must inherit to be compatible with GloMPO.
    Selectors are classes which return an optimizer and its configuration when asked by the manager. This selection will
    then be used to start a new optimizer. The full manager is supplied to the selector allowing sophisticated decisions
    to be designed.
    """

    def __init__(self,
                 *avail_opts: Union[Type[BaseOptimizer],
                                    Tuple[Type[BaseOptimizer], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
                 allow_spawn: Optional[Callable[['GloMPOManager'], bool]] = None):
        """
        Parameters
        ----------
        *avail_opts
            A set of optimizers available to the minimization. Elements may be:

            #. Subclasses (not instances) of :class:`.BaseOptimizer`.

            #. Tuples of:

               #. :class:`.BaseOptimizer` subclasses (not instances);

               #. Dictionary of kwargs sent to :meth:`.BaseOptimizer.__init__`, or :obj:`None`;

               #. Dictionary of kwargs sent to :meth:`.BaseOptimizer.minimize`, or :obj:`None`.

        allow_spawn
            Optional function sent to the selector which is called with the manager object as argument. If it returns
            :obj:`False` the manager will no longer spawn optimizers.

        Examples
        --------
        >>> DummySelector(OptimizerA, (OptimizerB, {'setting_a': 7}, None))

        The :code:`DummySelector` above may choose from two optimizers (:code:`OptimizerA` or :code:`OptimizerB`).
        :code:`OptimizerA` has no special configuration settings. :code:`OptimizerB` is configured with
        :code:`setting_a = 7` at initialisation, but no special arguments are needed for
        :meth:`OptimizerB.minimize() <.BaseOptimizer.minimize>` and thus :obj:`None` is sent in the last place of the
        tuple.

        >>> DummySelector(OptimizerA, allow_spawn=IterSpawnStop(50_000))

        In this case the selector will only spawn :code:`OptimizerA` optimizers but not allow any spawning after 50000
        function evaluations.
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
        """ Selects an optimizer to start from the available options.

        Parameters
        ----------
        manager
            :class:`.GloMPOManager` instance managing the optimization from which various attributes can be read.
        log
            :class:`.BaseLogger` instance containing the details and iteration history of every optimizer started thus
            far.
        slots_available
            Number of processes/threads the manager is allowed to start according to :attr:`.GloMPOManager.max_jobs` and
            the number of existing threads. GloMPO assumes that the selector will use this parameter to return an
            optimizer which requires fewer processes/threads than `slots_available`. If this is not possible then
            :obj:`None` is returned.

        Returns
        -------
        Union[Tuple[Type[BaseOptimizer], Dict[str, Any], Dict[str, Any]], None, bool]
            Optimizer class and configuration parameters: Tuple of optimizer class, dictionary of initialisation
            parameters, and dictionary of minmization parameters (see :meth:`__init__`). Manager will use this to
            initialise and start a new optimizer.

            :obj:`None` is returned in the case that no available optimizer configurations can satisfy the number of
            worker slots available.

            :obj:`False` is a special return which flags that the manager must never try and start another optimizer for
            the remainder of the optimization.
        """

    def __contains__(self, item):
        opts = (opt[0] for opt in self.avail_opts)
        return item in opts
