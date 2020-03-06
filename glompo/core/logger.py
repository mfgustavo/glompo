

from typing import *
from math import inf
import os
import yaml


class Logger:
    """ Stores progress of GloMPO optimizers. """
    def __init__(self):
        self.storage = {}

    def add_optimizer(self, opt_id: int, class_name: str, time_start: str):
        self.storage[opt_id] = OptimizerLogger(opt_id, class_name, time_start)

    def put_iteration(self, opt_id: int, i: int, x: Sequence[float], fx: float):
        self.storage[opt_id].append(i, x, fx)

    def put_metadata(self, opt_id: int, key: str, value: str):
        self.storage[opt_id].update_metadata(key, value)

    def put_message(self, opt_id: int, message: str):
        """ Optimizers can signal special messages to the optimizer during the optimization which can be saved to
            the log.
        """
        self.storage[opt_id].append_message(message)

    def get_history(self, opt_id, track: str = None) -> Union[List, Dict[int,
                                                                         Tuple[float, int, float, Sequence[float]]]]:
        extract = []
        if track:
            track_num = {"fx": 0,
                         "i_best": 1,
                         "fx_best": 2,
                         "x": 3}[track]
            for item in self.storage[opt_id].history.values():
                extract.append(item[track_num])
        else:
            extract = self.storage[opt_id].history
        return extract

    def save(self, name: str, opt_id: int = None):
        """ Saves the contents of the logger into yaml files. If an opt_id is provided only that optimizer will be
            saved using the provided name. Else all optimizers are saved by their opt_id numbers and type in a directory
            called name. """

        filename = name
        orig_dir = os.getcwd()
        if '/' in name:
            path, filename = name.rsplit('/', 1)
            os.makedirs(path, exist_ok=True)
            os.chdir(path)

        if opt_id:
            self._write_file(opt_id, filename)
        else:
            os.makedirs(filename, exist_ok=True)
            os.chdir(filename)
            for optimizer in self.storage:
                opt_id = self.storage[optimizer].metadata["Optimizer ID"]
                opt_type = self.storage[optimizer].metadata["Optimizer Type"]
                self._write_file(optimizer, f"{opt_id}_{opt_type}")
        os.chdir(orig_dir)

    def _write_file(self, opt_id, filename):
        with open(f"{filename}.yml", 'w') as file:
            data = {"DETAILS": self.storage[opt_id].metadata,
                    "MESSAGES": self.storage[opt_id].messages,
                    "ITERATION_HISTORY": self.storage[opt_id].history}
            yaml.dump(data, file, default_flow_style=False)


class OptimizerLogger:
    """ Stores history and meta data of a single optimzier started by GloMPO. """
    def __init__(self, opt_id: int, class_name: str, time_start: str):
        self.metadata = {"Optimizer ID": str(opt_id),
                         "Optimizer Type": class_name,
                         "Start Time": time_start}
        self.history = {}
        self.messages = []
        self.exit_cond = None

        self.fx_best = inf
        self.i_best = -1

    def update_metadata(self, key: str, value: str):
        """ Appends or overwrites given key-value pair in the stored optimizer metadata. """
        self.metadata[key] = value

    def append(self, i: int, x: Sequence[float], fx: float):
        if fx < self.fx_best:
            self.fx_best = fx
            self.i_best = i
        try:
            iter(x)
            ls = [float(num) for num in x]
            self.history[i] = [float(fx), int(self.i_best), float(self.fx_best), ls]
        except TypeError:
            ls = [float(num) for num in [x]]
            self.history[i] = [float(fx), int(self.i_best), float(self.fx_best), ls]

    def append_message(self, message):
        self.messages.append(message)
