""" Contains classes which save log information for GloMPO and its optimizers. """

import os
import warnings
from math import inf
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import yaml

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper as Dumper
try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as lines

    HAS_MATPLOTLIB = True
except (ModuleNotFoundError, ImportError):
    HAS_MATPLOTLIB = False

from glompo.common.helpers import FileNameHandler, LiteralWrapper, FlowList, literal_presenter, glompo_colors, \
    flow_presenter

__all__ = ("OptimizerLogger",)


class OptimizerLogger:
    """ Stores progress of GloMPO optimizers. """

    def __init__(self):
        self._storage: Dict[int, _OptimizerLogger] = {}
        yaml.add_representer(FlowList, flow_presenter, Dumper=Dumper)

    def __len__(self):
        return len(self._storage)

    def add_optimizer(self, opt_id: int, class_name: str, time_start: str):
        """ Adds a new optimizer data stream to the log. """
        self._storage[opt_id] = _OptimizerLogger(opt_id, class_name, time_start)

    def put_iteration(self, opt_id: int, i: int, f_call_overall: int, f_call_opt: int, x: Sequence[float], fx: float):
        """ Adds an iteration result to an optimizer data stream. """
        self._storage[opt_id].append(i, f_call_overall, f_call_opt, x, fx)

    def put_metadata(self, opt_id: int, key: str, value: str):
        """ Adds metadata about an optimizer. """
        self._storage[opt_id].update_metadata(key, value)

    def put_message(self, opt_id: int, message: str):
        """ Optimizers can signal special messages to the optimizer during the optimization which can be saved to
            the log.
        """
        self._storage[opt_id].append_message(message)

    def get_history(self, opt_id: int, track: Optional[str] = None) -> Union[List, Dict[int, Dict[str, float]]]:
        """ Returns a list of values for a given optimizer and track or returns the entire dictionary of all tracks
            if None.

            Parameters
            ----------
            opt_id: int
                Unique optimizer identifier.
            track: Optional[str] = None
                If specified returns only one series from the optimizer history. Available options:
                    - 'f_call_overall': The overall number of function evaluations used by all optimizers after each
                        iteration of opt_id,
                    - 'f_call_opt': The number of function evaluations used by opt_id after each of its iterations,
                    - 'fx': The function evaluations after each iteration,
                    - 'i_best': The iteration number at which the best function evaluation was located,
                    - 'fx_best': The best function evaluation value after each iteration,
                    - 'x': The task input values trialed at each iteration.
        """
        extract = []
        if track:
            for item in self._storage[opt_id].history.values():
                extract.append(item[track])
        else:
            extract = self._storage[opt_id].history
        return extract

    def get_metadata(self, opt_id, key: str) -> Any:
        """ Returns metadata of a given optimizer and key. """
        return self._storage[opt_id].metadata[key]

    def save_optimizer(self, name: str, opt_id: Optional[int] = None):
        """ Saves the contents of the logger into yaml files. If an opt_id is provided only that optimizer will be
            saved using the provided name. Else all optimizers are saved by their opt_id numbers and type in a directory
            called name.
        """
        with FileNameHandler(name) as filename:
            if opt_id:
                self._write_file(opt_id, filename)
            else:
                os.makedirs(filename, exist_ok=True)
                os.chdir(filename)

                digits = len(str(max(self._storage)))
                for optimizer in self._storage:
                    opt_id = int(self._storage[optimizer].metadata["Optimizer ID"])
                    opt_type = self._storage[optimizer].metadata["Optimizer Type"]
                    title = f"{opt_id:0{digits}}_{opt_type}"
                    self._write_file(optimizer, title)

    def save_summary(self, name: str):
        """ Generates a summary file containing the best found point of each optimizer and the reason for their
            termination. name is the path and filename of the summary file.
        """
        with FileNameHandler(name) as filename:
            sum_data = {}
            for optimizer in self._storage:
                opt_history = self.get_history(optimizer)

                i_tot = len(opt_history)
                x_best = None
                f_best = float('nan')
                f_calls = None
                if i_tot > 0 and opt_history[i_tot]['i_best'] > -1:
                    last = opt_history[i_tot]
                    i_best = last['i_best']
                    f_calls = last['f_call_opt']

                    best = opt_history[i_best]

                    x_best = FlowList(best['x'])
                    f_best = best['fx_best']
                sum_data[optimizer] = {'end_cond': self._storage[optimizer].metadata["End Condition"],
                                       'f_calls': f_calls,
                                       'f_best': f_best,
                                       'x_best': x_best}

            with open(filename, "w+") as file:
                yaml.dump(sum_data, file, Dumper=Dumper, default_flow_style=False, sort_keys=False)

    def plot_optimizer_trials(self, opt_id: Optional[int] = None):
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not present cannot create plots.", ImportWarning)
            return

        is_interactive = plt.isinteractive()
        if is_interactive:
            plt.ioff()

        opt_id = [opt_id] if opt_id else self._storage.keys()
        for opt in opt_id:
            x_all = self.get_history(opt, 'x')

            fig, ax = plt.subplots(figsize=(12, 8))
            fig: plt.Figure
            ax: plt.Axes

            ax.plot(x_all)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Parameter values as a function of optimizer iteration number')

            fig.savefig(f'opt{opt}_parms.png')
            plt.close(fig)

        if is_interactive:
            plt.ion()

    def plot_trajectory(self, title: str, log_scale: bool = False, best_fx: bool = False):
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not present cannot create plots.", ImportWarning)
            return

        is_interactive = plt.isinteractive()
        if is_interactive:
            plt.ioff()

        fig, ax = plt.subplots(figsize=(12, 8))
        fig: plt.Figure
        ax: plt.Axes

        leg_elements = [lines.Line2D([], [], ls='-', c='black', label='Optimizer Evaluations'),
                        lines.Line2D([], [], ls='', marker='x', c='black', label='Optimizer Killed'),
                        lines.Line2D([], [], ls='', marker='*', c='black', label='Optimizer Converged')]

        colors = glompo_colors()
        track = 'fx_best' if best_fx else 'fx'
        y_lab = "Best Function Evaluation" if best_fx else "Function Evaluation"
        for opt_id in self._storage.keys():
            f_calls = self.get_history(opt_id, 'f_call_overall')
            traj = self.get_history(opt_id, track)

            if log_scale:
                traj = np.sign(traj) * np.log10(np.abs(traj))
                stub = "fx_best" if best_fx else "fx"
                y_lab = f"sign({stub}) * log10(|{stub}|)"

            ax.plot(f_calls, traj, c=colors(opt_id))
            leg_elements.append(lines.Line2D([], [], ls='-', c=colors(opt_id),
                                             label=f"{opt_id}: {self.get_metadata(opt_id, 'Optimizer Type')}"))

            try:
                end_cond = self.get_metadata(opt_id, "End Condition")
                if "GloMPO Termination" in end_cond:
                    marker = 'x'
                elif "Optimizer convergence" in end_cond:
                    marker = '*'
                else:
                    marker = ''
                ax.plot(f_calls[-1], traj[-1], marker=marker, color='black')
            except (KeyError, IndexError):
                pass

        ax.set_xlabel('Function Calls')
        ax.set_ylabel(y_lab)
        ax.set_title("Optimizer function evaluations over time as a function of cumulative function calls.")

        # Apply Legend
        ax.legend(loc='upper right', handles=leg_elements, bbox_to_anchor=(1.35, 1))
        box = ax.get_position()
        ax.set_position([0.85 * box.x0, box.y0, 0.85 * box.width, box.height])

        fig.savefig(title)
        plt.close(fig)

        if is_interactive:
            plt.ion()

    def _write_file(self, opt_id, filename):
        yaml.add_representer(LiteralWrapper, literal_presenter, Dumper=Dumper)
        with open(f"{filename}.yml", 'w') as file:
            data = {"DETAILS": self._storage[opt_id].metadata,
                    "MESSAGES": self._storage[opt_id].messages,
                    "ITERATION_HISTORY": self._storage[opt_id].history}
            yaml.dump(data, file, Dumper=Dumper, default_flow_style=False, sort_keys=False)


class _OptimizerLogger:
    """ Stores history and meta data of a single optimizer started by GloMPO. """

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

    def append(self, i: int, f_call_overall: int, f_call_opt: int, x: Sequence[float], fx: float):
        """ Adds an optimizer iteration to the optimizer log. """
        if fx < self.fx_best:
            self.fx_best = fx
            self.i_best = i

        ls = None
        try:
            iter(x)
            ls = [float(num) for num in x]
        except TypeError:
            ls = [float(num) for num in [x]]
        finally:
            if i > 1:
                assert i == max(self.history) + 1, f"{i} == {max(self.history) + 1}"
            self.history[i] = {'f_call_overall': int(f_call_overall),
                               'f_call_opt': int(f_call_opt),
                               'fx': float(fx),
                               'i_best': int(self.i_best),
                               'fx_best': float(self.fx_best),
                               'x': FlowList(ls)}

    def append_message(self, message):
        """ Adds message to the optimizer history. """
        self.messages.append(message)
