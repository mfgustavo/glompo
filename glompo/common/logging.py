""" Contains class to split the logs of individual optimizers for easy processing. """

import logging
from pathlib import Path
from typing import Optional, Union


class SplitOptimizerLogs(logging.Filter):
    """ If this filter is applied to a Handler on the 'glompo.optimizers' logger it will automatically separate the
        single 'glompo.optimizers' logging stream into one for each individual optimizer.
    """

    def __init__(self, filepath: Union[Path, str] = "", propagate: bool = False,
                 formatter: Optional[logging.Formatter] = None):
        """
        Parameters
        ----------
        filepath: Union[Path, str] = ""
            Directory in which new log files will be located.
        propagate: bool = False
            If propagate is True then the filter will allow the message to pass through the filter allowing all
            glompo.optimizers logging to be simultaneously recorded together.
        formatter: Optional[logging.Formatter] = None
            Formatting to be applied in the new logs. If not supplied the logging module default is used.

        Examples
        --------
            formatter = logging.Formatter("%(levelname)s : %(name)s : %(processName)s :: %(message)s")

            # Adds individual handlers for each optimizer created
            # Format for the new handlers is set by formatter
            # Propagate=True sends the message on to opt_handler which in this case is stdout
            opt_filter = SplitOptimizerLogs("diverted_logs", propagate=True, formatter=formatter)
            opt_handler = logging.StreamHandler(sys.stdout)
            opt_handler.addFilter(opt_filter)
            opt_handler.setFormatter(formatter)

            # Messages of the INFO level will propogate to stdout
            opt_handler.setLevel('INFO')

            logging.getLogger("glompo.optimizers").addHandler(opt_handler)

            # The level for the handlers made in SplitOptimizerLogs is set at the higher level.
            # Here DEBUG level messages will be logged to the files even though INFO level propagates to the console
            logging.getLogger("glompo.optimizers").setLevel('DEBUG')
        """
        super().__init__('')
        self.opened = set()
        self.filepath = Path(filepath) if filepath else Path.cwd()
        self.filepath.mkdir(parents=True, exist_ok=True)
        self.propagate = int(propagate)
        self.fomatter = formatter

    def filter(self, record: logging.LogRecord) -> int:

        opt_id = int(record.name.replace("glompo.optimizers.opt", ""))

        if opt_id not in self.opened:
            self.opened.add(opt_id)

            if self.fomatter:
                message = self.fomatter.format(record)
            else:
                message = record.getMessage()

            with (self.filepath / f"optimizer_{opt_id}.log").open('w+') as file:
                file.write(f"{message}\n")

            handler = logging.FileHandler(self.filepath / f"optimizer_{opt_id}.log", 'a')

            if self.fomatter:
                handler.setFormatter(self.fomatter)

            logger = logging.getLogger(record.name)
            logger.addHandler(handler)

        return self.propagate
