

import logging
import os
from typing import *


class SplitOptimizerLogs(logging.Filter):

    """ If this filter is applied to a Handler on the 'glompo.optimizers' logger it will automatically separate the
        single 'glompo.optimizers' logging stream into one for each individual optimizer.
    """

    def __init__(self, filepath: str = "", propagate: bool = False, formatter: Optional[logging.Formatter] = None):
        """
        Parameters
        ----------
        filepath: str = ""
            Directory in which new log files will be located.
        propagate: bool = False
            If propagate is True then the filter will allow the message to pass through the filter allowing all
            glompo.optimizers logging to be simultaneously recorded together.
        formatter: Optional[logging.Formatter] = None
            Formatting to be applied in the new logs. If not supplied the logging module default is used.

        Examples
        --------
            formatter = logging.Formatter("%(levelname)s : %(name)s : %(processName)s :: %(message)s")

            filter = logging.Filter('glompo.optimizers')
            split_filter = SplitOptimizerLogs("diverted_logs", formatter=formatter)

            handler = logging.StreamHandler(sys.stdout)
            opt_handler.setFormatter(formatter)
            opt_handler.addFilter(opt_filter)
            opt_handler.addFilter(opt_split_filter)
            opt_handler.setLevel('DEBUG')

            logger = logging.getLogger('glompo')
            logger.addHandler(opt_handler)
            logger.setLevel('DEBUG')

            manager = GloMPOManager(...)
            manager.start_manager()
        """
        self.opened = set()
        self.filepath = filepath + '/' if filepath else ""
        self.propagate = int(propagate)
        self.fomatter = formatter

    def filter(self, record: logging.LogRecord) -> int:
        opt_id = int(record.name.replace("glompo.optimizers.opt", ""))

        if opt_id not in self.opened:
            self.opened.add(opt_id)

            if self.filepath:
                try:
                    os.makedirs(f"{self.filepath}")
                except FileExistsError:
                    pass

            if self.fomatter:
                message = self.fomatter.format(record)
            else:
                message = record.getMessage()

            with open(f"{self.filepath}optimizer_{opt_id}.log", 'w+') as file:
                file.write(f"{message}\n")

            handler = logging.FileHandler(f"{self.filepath}optimizer_{opt_id}.log", 'a')

            if self.fomatter:
                handler.setFormatter(self.fomatter)

            logger = logging.getLogger(record.name)
            logger.addHandler(handler)

        return self.propagate
