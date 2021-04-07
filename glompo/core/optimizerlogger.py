""" Contains classes which save log information for GloMPO and its optimizers. """

import warnings
from math import inf
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, overload

import numpy as np
import yaml

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper
try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as lines

    HAS_MATPLOTLIB = True
except (ModuleNotFoundError, ImportError):
    HAS_MATPLOTLIB = False

from ..common.helpers import FlowList, glompo_colors, flow_presenter, rolling_best

__all__ = ("OptimizerLogger",)


class OptimizerLogger:
    """ Stores progress of GloMPO optimizers. """

    def __init__(self):
        self._storage: Dict[int, _OptimizerLogger] = {}
        self._best_iter = {'opt_id': 0, 'x': [], 'fx': float('inf')}
        yaml.add_representer(FlowList, flow_presenter, Dumper=Dumper)

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        """ Returns an iterable of optimizer IDs. """
        return iter(self._storage)

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __getitem__(self, item) -> '_OptimizerLogger':
        """ Returns an individual optimizer log. """
        return self._storage[item]

    def items(self):
        """ Returns an iterable of tuples of optimizer IDs and logs. """
        return self._storage.items()

    def keys(self):
        """ Returns an iterable of optimizer IDs. """
        return self._storage.keys()

    def values(self):
        """ Returns an iterable of individual optimizer logs. """
        return self._storage.values()

    @property
    def best_iter(self) -> Dict[str, Any]:
        return self._best_iter

    def add_optimizer(self, opt_id: int, class_name: str, time_start: str):
        """ Adds a new optimizer data stream to the log. """
        self[opt_id] = _OptimizerLogger(opt_id, class_name, time_start)

    def put_iteration(self, opt_id: int, i: int, f_call_overall: int, f_call_opt: int, x: Sequence[float], fx: float):
        """ Adds an iteration result to an optimizer data stream. """
        if fx < self._best_iter['fx']:
            self._best_iter['opt_id'] = opt_id
            self._best_iter['x'] = x
            self._best_iter['fx'] = fx

        self[opt_id].append(i, f_call_overall, f_call_opt, x, fx)

    def put_metadata(self, opt_id: int, key: str, value: str):
        """ Adds metadata about an optimizer. """
        self[opt_id].update_metadata(key, value)

    def put_message(self, opt_id: int, message: str):
        """ Optimizers can signal special messages to the optimizer during the optimization which can be saved to
            the log.
        """
        self[opt_id].append_message(message)

    @overload
    def get_history(self, opt_id: int) -> Dict[int, Dict[str, float]]:
        ...

    @overload
    def get_history(self, opt_id: int, track: str) -> List:
        ...

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
                    - 'x': The task input values trialed at each iteration.
        """
        if track:
            extract = [item[track] for item in self[opt_id].history.values()]
        else:
            extract = self[opt_id].history
        return extract

    def get_metadata(self, opt_id, key: str) -> Any:
        """ Returns metadata of a given optimizer and key. """
        return self[opt_id].metadata[key]

    def save_summary(self, path: Union[Path, str]):
        """ Generates a summary file containing the best found point of each optimizer and the reason for their
            termination.
        """
        sum_data = {}
        for opt_id, opt_log in self.items():
            i_tot = len(opt_log.history)

            i_best = None
            x_best = None
            f_best = float('nan')
            f_calls = None

            if i_tot > 0 and opt_log.i_best > -1:
                i_best = opt_log.i_best
                x_best = FlowList(opt_log.x_best)
                f_best = opt_log.fx_best
                f_calls = opt_log.history[i_tot]['f_call_opt']

            sum_data[opt_id] = {**opt_log.metadata,
                                **{'f_calls': f_calls,
                                   'i_best': i_best,
                                   'f_best': f_best,
                                   'x_best': x_best},
                                'messages': opt_log.messages}
            del sum_data[opt_id]['opt_id']

        with Path(path).open("w+") as file:
            yaml.dump(sum_data, file, Dumper=Dumper, default_flow_style=False, sort_keys=False)

    def plot_optimizer_trials(self, path: Optional[Path] = None, opt_id: Optional[int] = None):
        """ Generates plots for each optimizer in the log of each trialed parameter value as a function of optimizer
            iterations.
        """

        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not present cannot create plots.", ImportWarning)
            return

        is_interactive = plt.isinteractive()
        if is_interactive:
            plt.ioff()

        opt_id = [opt_id] if opt_id else self.keys()
        for opt in opt_id:
            x_all = self.get_history(opt, 'x')

            fig, ax = plt.subplots(figsize=(12, 8))
            fig: plt.Figure
            ax: plt.Axes

            ax.plot(x_all)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Parameter values as a function of optimizer iteration number')

            name = f'opt{opt}_parms.png' if path is None else Path(path, f'opt{opt}_parms.png')
            fig.savefig(name)
            plt.close(fig)

        if is_interactive:
            plt.ion()

    def plot_trajectory(self, title: Union[Path, str], log_scale: bool = False, best_fx: bool = False):
        """ Generates a plot of each optimizer function values versus the overall function evaluation number.

            Parameters
            ----------
            title: Union[Path, str]
                Path to file to which the plot should be saved.
            log_scale: bool = False
                If True the function evaluations will be converted to base 10 log values.
            best_fx: bool = False
                If True the best function evaluation see thus far of each optimizer will be plotted rather than the
                function evaluation at the matching evaluation number.
        """

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
        y_lab = "Best Function Evaluation" if best_fx else "Function Evaluation"
        for opt_id in self:
            f_calls = self.get_history(opt_id, 'f_call_overall')
            traj = self.get_history(opt_id, 'fx')
            if best_fx:
                traj = rolling_best(traj)

            if log_scale:
                traj = np.sign(traj) * np.log10(np.abs(traj))
                stub = "fx_best" if best_fx else "fx"
                y_lab = f"sign({stub}) * log10(|{stub}|)"

            ax.plot(f_calls, traj, ls='-', marker='.', c=colors(opt_id))
            leg_elements.append(lines.Line2D([], [], ls='-', c=colors(opt_id),
                                             label=f"{opt_id}: {self.get_metadata(opt_id, 'opt_type')}"))

            try:
                end_cond = self.get_metadata(opt_id, "end_cond")
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


class _OptimizerLogger:
    """ Stores history and meta data of a single optimizer started by GloMPO. """

    def __init__(self, opt_id: int, class_name: str, time_start: str):
        self.metadata = {"opt_id": str(opt_id),
                         "opt_type": class_name,
                         "t_start": time_start}
        self.history = {}
        self.messages = []
        self.exit_cond = None

        self.fx_best = inf
        self.i_best = -1
        self.f_call_best = -1
        self.x_best = []

    def __len__(self):
        return len(self.history)

    def update_metadata(self, key: str, value: str):
        """ Appends or overwrites given key-value pair in the stored optimizer metadata. """
        self.metadata[key] = value

    def append(self, i: int, f_call_overall: int, f_call_opt: int, x: Sequence[float], fx: float):
        """ Adds an optimizer iteration to the optimizer log. """
        if fx < self.fx_best:
            self.fx_best = fx
            self.i_best = i
            self.x_best = x
            self.f_call_best = f_call_opt

        ls = None
        try:
            iter(x)
            ls = [float(num) for num in x]
        except TypeError:
            ls = [float(num) for num in [x]]
        finally:
            if i > 1:
                assert i == max(
                    self.history) + 1, f"Opt {self.metadata['opt_id']}: {i} != {max(self.history) + 1}"
            self.history[i] = {'f_call_overall': int(f_call_overall),
                               'f_call_opt': int(f_call_opt),
                               'fx': float(fx),
                               'x': FlowList(ls)}

    def append_message(self, message):
        """ Adds message to the optimizer history. """
        self.messages.append(message)
