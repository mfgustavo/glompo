""" Contains classes which save log information for GloMPO and its optimizers. """
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import tables as tb

from ..common.helpers import deepsizeof, glompo_colors, rolling_min
from ..common.namedtuples import IterationResult
from ..common.wrappers import needs_optional_package

try:
    import matplotlib.pyplot as plt
    import matplotlib.lines as lines
    import dill
except (ModuleNotFoundError, ImportError):
    pass

__all__ = ("BaseLogger",
           "FileLogger")


class BaseLogger:
    """ Holds iteration results in memory for faster access.

    Attributes
    ----------
    """

    @classmethod
    @needs_optional_package('dill')
    def checkpoint_load(cls, path: Union[Path, str]):
        """ Construct a new BaseLogger from the attributes saved in the checkpoint file located at path. """
        opt_log = cls.__new__(cls)

        with Path(path).open('rb') as file:
            state = dill.load(file)

        for var, val in state.items():
            opt_log.__setattr__(var, val)

        return opt_log

    def __init__(self, build_traj_plot: bool, *args, **kwargs):
        self._f_counter = 0  # Total number of evaluations accepted
        self._best_iters = {0: {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0}}
        self._best_iter = {'opt_id': 0, 'x': [], 'fx': float('inf'), 'type': '', 'call_id': 0}
        self._max_eval = -float('inf')
        self._storage = {}
        self.build_traj_plot = build_traj_plot
        self._figure_data = {}

    def __contains__(self, item) -> bool:
        """ Returns True if the optimizer is being recorded in memory. """
        return item in self._storage

    def __len__(self) -> int:
        """ Returns the total number of function evaluations saved in the log. """
        return self._f_counter

    @property
    def n_optimizers(self) -> int:
        """ Returns the number of optimizer in the log. """
        return len(self._storage)

    @property
    def largest_eval(self) -> float:
        """ Returns the largest (finite) function evaluation processed thus far. """
        return self._max_eval

    def len(self, opt_id: int) -> int:
        """ Returns the number of function evaluations associated with optimizer opt_id. """
        if self.has_iter_history(opt_id):
            return len(self._storage[opt_id]['fx'])
        return 0

    def add_optimizer(self, opt_id: int, opt_type: str, t_start: datetime.datetime):
        """ Creates a space in memory for a new optimizer. """
        self._best_iters[opt_id] = {'opt_id': opt_id, 'x': [], 'fx': float('inf'), 'type': opt_type, 'call_id': 0}
        self._storage[opt_id] = {'metadata': {'opt_id': opt_id, 'opt_type': opt_type, 't_start': t_start},
                                 'messages': []}

    def add_iter_history(self, opt_id: int, extra_headers: Optional[Dict[str, tb.Col]] = None):
        """ Extends iteration history with all the columns required, including possible detailed calls. """
        headers = ['call_id', 'x', 'fx']
        if extra_headers:
            headers += [*extra_headers]

        for k in headers:
            self._storage[opt_id][k] = []

    def has_iter_history(self, opt_id: int) -> bool:
        """ Returns True if an iteration history table has been constructed for optimizer opt_id. """
        return opt_id in self._storage and 'fx' in self._storage[opt_id]

    def clear_cache(self, opt_id: Optional[int] = None):
        """ Removes all data associated with opt_id form memory. The data is NOT cleared if a summary trajectory plot
            has been configured.
        """
        if self.build_traj_plot:  # Data not cleared if a summary trajectory image has been requested.
            return

        to_del = [opt_id] if opt_id else [*self._storage.keys()]
        for key in to_del:
            del self._storage[key]

    def put_metadata(self, opt_id: int, key: str, value: Any):
        """ Adds metadata to storage. """
        self._storage[opt_id]['metadata'][key] = value

    def put_manager_metadata(self, key: str, value: Any):
        pass

    def put_message(self, opt_id: int, message: str):
        self._storage[opt_id]['messages'].append(message)

    def put_iteration(self, iter_res: IterationResult):
        """ Records function evaluation in memory. """
        self._f_counter += 1

        if iter_res.fx < self._best_iters[iter_res.opt_id]['fx']:
            self._best_iters[iter_res.opt_id]['x'] = iter_res.x
            self._best_iters[iter_res.opt_id]['fx'] = iter_res.fx
            self._best_iters[iter_res.opt_id]['call_id'] = self._f_counter

            if iter_res.fx < self._best_iter['fx']:
                self._best_iter = self._best_iters[iter_res.opt_id]

        if iter_res.fx > self._max_eval and np.isfinite(iter_res.fx):
            self._max_eval = iter_res.fx

        for k, v in zip(self._storage[iter_res.opt_id],
                        (None, None, self._f_counter, iter_res.x, iter_res.fx, *iter_res.extras)):
            if k in ('metadata', 'messages'):
                continue
            self._storage[iter_res.opt_id][k].append(v)

    def get_best_iter(self, opt_id: Optional[int] = None) -> Dict[str, Any]:
        """ Returns the overall best record in history if opt_id is not provided. If it is, the best iteration
            of the corresponding optimizer is returned.
        """
        if opt_id:
            return self._best_iters[opt_id]
        return self._best_iter

    def get_history(self, opt_id: int, track: str) -> List:
        """ Returns all the evaluations associated with optimizer opt_id. track refers to the column of interest (e.g.
            'call_id', 'x', 'fx').
        """
        if self.has_iter_history(opt_id):
            return self._storage[opt_id][track]
        return []

    def get_metadata(self, opt_id: int, key: str) -> Any:
        """ Returns a piece of metadata in memory. """
        return self._storage[opt_id]['metadata'][key]

    @needs_optional_package('matplotlib')
    def plot_optimizer_trials(self, path: Optional[Path] = None, opt_id: Optional[int] = None):
        """ Generates plots for each optimizer in the log of each trialed parameter value as a function of optimizer
            iterations.
        """
        is_interactive = plt.isinteractive()
        if is_interactive:
            plt.ioff()

        opt_ids = [opt_id] if opt_id else range(1, self.n_optimizers + 1)

        for opt in opt_ids:
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

    @needs_optional_package('matplotlib')
    def plot_trajectory(self, title: Union[Path, str], log_scale: bool = False, best_fx: bool = False):
        """ Generates a plot of each optimizer function values versus the overall function evaluation number.

            Parameters
            ----------
            title
                Path to file to which the plot should be saved.
            log_scale
                If True the function evaluations will be converted to base 10 log values.
            best_fx
                If True the best function evaluation see thus far of each optimizer will be plotted rather than the
                function evaluation at the matching evaluation number.
        """
        is_interactive = plt.isinteractive()
        if is_interactive:
            plt.ioff()

        fig, ax = plt.subplots(figsize=(12, 8))
        fig: plt.Figure
        ax: plt.Axes

        leg_elements = [lines.Line2D([], [], ls='-', c='black', label='Optimizer Evaluations'),
                        lines.Line2D([], [], ls='', marker='x', c='black', label='Optimizer Killed'),
                        lines.Line2D([], [], ls='', marker='s', c='black', label='Optimizer Crashed'),
                        lines.Line2D([], [], ls='', marker='*', c='black', label='Optimizer Converged')]

        colors = glompo_colors()
        y_lab = "Best Function Evaluation" if best_fx else "Function Evaluation"
        for opt_id in range(1, self.n_optimizers + 1):
            f_calls = self.get_history(opt_id, 'call_id')
            traj = self.get_history(opt_id, 'fx')
            if best_fx:
                traj = rolling_min(traj)

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
                elif "Optimizer convergence" in end_cond or "Normal termination" in end_cond:
                    marker = '*'
                elif "Error termination" in end_cond or "Traceback" in end_cond:
                    marker = 's'
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

    def flush(self, opt_id: Optional[int] = None):
        pass

    def open(self, path: Union[Path, str], mode: str, checksum: str):
        pass

    def close(self):
        """ Remove all records from memory. """
        self.clear_cache()

    @needs_optional_package('dill')
    def checkpoint_save(self, path: Union[Path, str] = '', block: Optional[Sequence[str]] = None):
        """ Saves the state of the logger, suitable for resumption, during a checkpoint.

            Parameters
            ----------
            path
                Directory in which to dump the generated files.
            block
                Iterable of class attributes which should not be included in the log.
        """
        block = block if block else []
        block += ['n_optimizers', 'largest_eval']
        dump_variables = {}
        for var in dir(self):
            if '__' not in var and not callable(getattr(self, var)) and all([var != b for b in block]):
                dump_variables[var] = getattr(self, var)

        with Path(path, 'opt_log').open('wb') as file:
            dill.dump(dump_variables, file)


# noinspection PyProtectedMember
class FileLogger(BaseLogger):
    """ Extends the BaseLogger to write progress of GloMPO optimizers to disk in HDF5 format through PyTables.
        Results of living optimizers are still held in memory to optimizer hunting.
    """

    def __init__(self,
                 n_parms: int,
                 expected_rows: int,
                 build_traj_plot: bool):
        """ Setups and opens the logfile.

            Parameters
            ----------
            n_parms
                Number of parameters in the domain of the optimization problem.
            expected_rows
                Estimated number of rows in each optimizer log file. Estimated by GloMPOManager based on convergence
                settings and dimensionality of the optimization task.
            build_traj_plot
                Flag the logger to hold trajectories in memory to construct the summary image.
        """
        super().__init__(build_traj_plot)
        self.pytab_file = None

        self.expected_rows = expected_rows
        self.n_task_dims = n_parms

        self._o_counter = 0  # Total number of optimizers started
        self._writing_chunk = {}  # Iterations are written to disk in chunks save time
        self._est_iter_size = 0  # Estimated size of a single iteration result
        self._groups = {}  # In memory address to pytables_file groups (expensive otherwise)
        self._tables = {}  # In memory address to pytables_file tables (expensive otherwise)

    def __contains__(self, opt_id: int) -> bool:
        """ Returns True if a group exists in the HDF5 file for the optimizer with ID opt_id """
        return f'/optimizer_{opt_id}' in self.pytab_file

    @property
    def n_optimizers(self):
        """ Returns the number of optimizers in the log. """
        return self._o_counter

    def len(self, opt_id: int) -> int:
        """ Returns the number of function evaluations associated with optimizer opt_id. """
        try:
            return super().len(opt_id)
        except KeyError:
            if self.has_iter_history(opt_id):
                return self._tables[opt_id].nrows
            return 0

    def add_optimizer(self, opt_id: int, opt_type: str, t_start: datetime.datetime):
        """ Creates a new optimizer H5 file and memory logger. """
        super().add_optimizer(opt_id, opt_type, t_start)
        group = self.pytab_file.create_group(where='/',
                                             name=f'optimizer_{opt_id}')
        self.pytab_file.create_vlarray(where=f'/optimizer_{opt_id}',
                                       name='messages',
                                       atom=tb.VLUnicodeAtom(),
                                       title="Messages Generated by Optimizer",
                                       expectedrows=3)
        self._writing_chunk[opt_id] = []
        self._groups[opt_id] = group

        for key, val in zip(('opt_id', 'opt_type', 't_start'),
                            (opt_id, opt_type, t_start)):
            group._v_attrs[key] = val

        self._o_counter += 1

    def add_iter_history(self, opt_id: int, extra_headers: Optional[Dict[str, tb.Col]] = None):
        """ Creates an iteration history table in the H5 file. """
        super().add_iter_history(opt_id, {})  # Do not hold extras in memory if file in use.
        headers = {'call_id': tb.UInt32Col(pos=-3),
                   'x': tb.Float64Col(shape=self.n_task_dims, pos=-2),
                   'fx': tb.Float64Col(pos=-1)}

        if extra_headers:
            headers = {**headers, **extra_headers}

        table = self.pytab_file.create_table(where=f'/optimizer_{opt_id}',
                                             name='iter_hist',
                                             description=headers,
                                             title="Iteration History",
                                             expectedrows=self.expected_rows)
        self._tables[opt_id] = table

    def has_iter_history(self, opt_id: int) -> bool:
        """ Returns True if an iteration history table has been constructed for optimizer opt_id. """
        return opt_id in self._tables

    def put_iteration(self, iter_res: IterationResult):
        """ Records a function evaluation to memory. """
        try:
            super().put_iteration(iter_res)  # Increment f_counter and update best_iters
        except KeyError:
            pass

        if self._est_iter_size == 0:
            self._est_iter_size = deepsizeof(iter_res)

        self._writing_chunk[iter_res.opt_id].append(
            [(self._f_counter, iter_res.x, iter_res.fx, *iter_res.extras)])

        if self._est_iter_size * sum((len(c) for c in self._writing_chunk.values())) > 100_000_000:  # Flush every 100MB
            self.flush(iter_res.opt_id)

    def put_metadata(self, opt_id: int, key: str, value: Any):
        """ Adds metadata about an optimizer. """
        try:
            super().put_metadata(opt_id, key, value)
        except KeyError:
            pass
        self._get_group(opt_id)._v_attrs[key] = value

    def put_manager_metadata(self, key: str, value: Any):
        """ Records optimization settings and history information (similar to that in glompo_manager_log.yml) into the
            H5 file.
        """
        self.pytab_file.root._v_attrs[key] = value

    def put_message(self, opt_id: int, message: str):
        """ Optimizers can signal special messages to the optimizer during the optimization which can be saved to
            the log.
        """
        super().put_message(opt_id, message)
        table = self._get_group(opt_id)['messages']
        table.append(message)
        table.flush()

    def get_metadata(self, opt_id, key: str) -> Any:
        """ Returns metadata of a given optimizer and key. """
        try:
            return super().get_metadata(opt_id, key)
        except KeyError:
            return self._get_group(opt_id)._v_attrs[key]

    def get_history(self, opt_id: int, track: str) -> List:
        """ Returns data from the evaluation history of optimizer opt_id.

            Parameters
            ----------
            opt_id
                Unique optimizer identifier.
            track
                Column name to return. Any column name in the logfile can be used. The following are always present:
                    - 'call_id'
                        The overall evaluation number across all function calls.
                    - 'x'
                        Input vectors evaluated by the optimizer.
                    - 'fx'
                        The function response for each iteration.
        """
        try:
            return super().get_history(opt_id, track)
        except KeyError:
            if self.has_iter_history(opt_id):
                self.flush(opt_id)
                table = self._get_table(opt_id)
                return table.col(track)

            return []

    def _get_group(self, opt_id: int) -> tb.Group:
        """ Returns the the tables.Group object for optimizer opt_id"""
        if opt_id not in self._groups:
            self._groups[opt_id] = self.pytab_file.get_node('/', f'optimizer_{opt_id}')
        return self._groups[opt_id]

    def _get_table(self, opt_id: int) -> tb.Table:
        """ Returns the the tables.Table object for optimizer opt_id"""
        if not self.has_iter_history(opt_id):
            self._tables[opt_id] = self.pytab_file.get_node('/', f'optimizer_{opt_id}/iter_hist')
        return self._tables[opt_id]

    def flush(self, opt_id: Optional[int] = None):
        """ Writes iterations held in chunks to disk. If opt_id is provided then the corresponding
            optimizer is closed, else all optimizers are closed in this way.
        """
        opt_ids = [opt_id] if opt_id else self._writing_chunk.keys()

        for o in opt_ids:
            if len(self._writing_chunk[o]) > 0:
                self.put_metadata(o, 'best_iter', self._best_iters[o])
                table = self._get_table(o)
                table.append(self._writing_chunk[o])
                self._writing_chunk[o] = []
                table.flush()

    def clear_cache(self, opt_id: Optional[int] = None):
        """ Clears information held in the cache for hunting purposes. If opt_id is provided then the corresponding
            optimizer is closed, else all optimizers are closed in this way.
        """
        opt_ids = [opt_id] if opt_id else range(1, self.n_optimizers + 1)
        for o in opt_ids:
            if o in self._writing_chunk:
                self.flush(o)
                del self._writing_chunk[o]

            if super().__contains__(o):
                super().clear_cache(o)

    def open(self, path: Union[Path, str], mode: str, checksum: str):
        """ Opens or creates the H5 file.

            Parameters
            ----------
            path
                File path in which to construct the logfile.
            mode
                The open mode of the file. 'w' and 'a' modes are supported.
            checksum
                Unique checksum value generated by GloMPOManager and stored in checkpoints and the logfile. When a
                checkpoint is loaded, GloMPO will confirm a match between the checksum value in the checkpoint and in
                the logfile before using it.
        """
        self.pytab_file = tb.open_file(str(path), mode, filters=tb.Filters(1, 'blosc'))
        self.pytab_file.root._v_attrs.checksum = checksum
        if mode == 'a':
            self._groups = {int(g._v_name.split('_')[1]): g for g in self.pytab_file.iter_nodes('/', 'Group')}
            self._tables = {int(t._v_pathname.split('/')[1].split('_')[1]): t for t in
                            self.pytab_file.walk_nodes('/', 'Table')}

    def close(self):
        """ Remove from memory, flush to file and close the file. """
        self.pytab_file.root._v_attrs.opts_started = self._o_counter
        self.pytab_file.root._v_attrs.f_counter = self._f_counter
        self.pytab_file.root._v_attrs.best_iters = self._best_iters
        self.pytab_file.root._v_attrs.best_iter = self._best_iter
        self.pytab_file.root._v_attrs.max_eval = self._max_eval

        self.flush()
        self.pytab_file.flush()
        self.pytab_file.close()

    def checkpoint_save(self, path: Union[Path, str] = '', block: Optional[Sequence[str]] = None):
        """ Saves the state of the logger, suitable for resumption, during a checkpoint. Path is a directory in which to
            dump the generated files.
        """
        super().checkpoint_save(path, ['pytab_file', '_tables', '_groups'])
