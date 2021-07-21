""" Contains the GloMPOScope class which is a useful extension allowing a user to visualize GloMPO's behaviour. """
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import matplotlib.animation as ani
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np

from ..common.helpers import glompo_colors
from ..common.wrappers import catch_user_interrupt, decorate_all_methods, needs_optional_package

try:
    import dill
except ModuleNotFoundError:
    pass

__all__ = ("GloMPOScope",)


class MyFFMpegWriter(ani.FFMpegWriter):
    """ Overwrites a method in the matplotlib.animation.FFMpegWriter class which caused it to hang during movie
        generation.
    """

    def cleanup(self):
        """ Clean-up and collect the process used to write the movie file. """
        self._frame_sink().close()


@decorate_all_methods(catch_user_interrupt)
class GloMPOScope:
    """ Constructs and records the dynamic plotting of optimizers run in parallel. """

    def __init__(self,
                 x_range: Union[Tuple[float, float], int, None] = 300,
                 y_range: Optional[Tuple[float, float]] = None,
                 log_scale: bool = False,
                 record_movie: bool = False,
                 interactive_mode: bool = False,
                 events_per_flush: int = 10,
                 elitism: bool = False,
                 writer_kwargs: Optional[Dict[str, Any]] = None,
                 movie_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initializes the plot and movie recorder.

        Parameters
        ----------
        x_range: Union[Tuple[float, float], int, None] = 300
            If None is provided the x-axis will automatically and continuously rescale from zero as the number of
            function evaluations increases.
            If a tuple of the form (min, max) is provided then the x-axis will be fixed to this range.
            If an integer is provided then the plot will only show the last x_range evaluations and discard earlier
            points. This is useful to make differences between optimizers visible in the late stage and also keep the
            scope operating at an adequate speed.
        y_range: Optional[Tuple[float, float]] = None
            Sets the y-axis limits of the plot, default is an empty tuple which leads the plot to automatically and
            constantly rescale the axis.
        log_scale: bool = False
            If True, the base 10 logarithm of y values are displayed on the scope. This can be used in conjunction
            with the y_range option and will be interpreted in the opt_log-scale.
        record_movie: bool = False
            If True then a matplotlib.animation.FFMpegWriter instance is created to record the plot.
        interactive_mode: bool = False
            If True the plot is visible on screen during the optimization.
        events_per_flush: int = 10
            The number of 'events' or updates and changes to the scope before the changes are flushed and the plot is
            redrawn. A lower number provides a smoother visualisation but is expensive and, if recorded,
            takes a larger amount of space.
        elitism: bool = False
            If True the scope will only display values which improve on the best optimizer value.
        writer_kwargs: Optional[Dict[str, Any]] = None
            Optional dictionary of arguments to be sent to the initialisation of the matplotlib.animation.FFMpegWriter
            class.
        movie_kwargs: Optional[Dict[str, Any]] = None
            Optional dictionary of arguments to be sent to matplotlib.animation.FFMpegWriter.setup().
        """
        self.logger = logging.getLogger('glompo.scope')
        self._dead_streams: Set[int] = set()
        self.n_streams = 0
        self.x_max = 0
        self.log_scale = log_scale
        self.events_per_flush = events_per_flush
        self.elitism = bool(elitism)
        self._event_counter = 0
        self.color_map = glompo_colors()
        self.interactive_mode = interactive_mode
        self.record_movie = record_movie
        self.is_setup = False

        plt.ion() if interactive_mode else plt.ioff()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title("GloMPO Scope")
        self.ax.set_xlabel("Total Function Calls")
        if self.log_scale:
            self.ax.set_ylabel("Log10(Function Evaluation)")
            self.logger.debug("Using log scale error values")
        else:
            self.ax.set_ylabel("Function Evaluation")
            self.logger.debug("Using linear scale error values")

        # Create custom legend
        self.leg_elements = [lines.Line2D([], [], ls='-', c='black', label='Optimizer Evaluations'),
                             lines.Line2D([], [], ls='', marker='x', c='black', label='Optimizer Killed'),
                             lines.Line2D([], [], ls='', marker='*', c='black', label='Optimizer Converged'),
                             lines.Line2D([], [], ls='', marker='s', c='black', label='Optimizer Crashed'),
                             lines.Line2D([], [], ls='', marker='4', c='black', label='Optimizer Paused'),
                             lines.Line2D([], [], ls='', marker='|', c='black', label='Checkpoint')]

        self.ax.legend(loc='upper right', handles=self.leg_elements, bbox_to_anchor=(1.35, 1))

        self.opt_streams: Dict[int, plt.Axes.plot] = {}
        self.gen_streams: Dict[str, plt.Axes.plot] = {
            'opt_kill': self.ax.plot([], [], ls='', marker='x', color='black', zorder=500)[0],
            'opt_norm': self.ax.plot([], [], ls='', marker='*', color='black', zorder=500)[0],
            'opt_crash': self.ax.plot([], [], ls='', marker='s', color='black', zorder=500)[0],
            'pause': self.ax.plot([], [], ls='', marker='4', color='black', zorder=500)[0],
            'chkpt': self.ax.plot([], [], ls='', marker='|', color='black', zorder=500)[0]}

        # Setup and shrink axis position to fit legend
        box = self.ax.get_position()
        self.ax.set_position([0.85 * box.x0, box.y0, 0.85 * box.width, box.height])

        # Setup axis limits
        self.truncated = None
        if x_range is None:
            self.ax.set_autoscalex_on(True)
        elif isinstance(x_range, tuple):
            if x_range[0] >= x_range[1]:
                self.logger.critical("Cannot parse x_range, min >= max.")
                self.close_fig()
                raise ValueError(f"Cannot parse x_range = {x_range}. Min must be less than and not equal to max.")
            self.ax.set_xlim(x_range[0], x_range[1])
        elif isinstance(x_range, int):
            if x_range < 2:
                self.logger.critical("Cannot parse x_range, x_range < 2")
                self.close_fig()
                raise ValueError(f"Cannot parse x_range = {x_range}. Value larger than 1 required.")
            self.truncated = x_range
        else:
            self.logger.critical("Cannot parse x_range. Unsupported type. None, int or tuple expected.")
            self.close_fig()
            raise TypeError(f"Cannot parse x_range = {x_range}. Only int, NoneType and tuple can be used.")

        if isinstance(y_range, tuple):
            if y_range[0] >= y_range[1]:
                self.logger.critical("Cannot parse y_range, min >= max")
                self.close_fig()
                raise ValueError(f"Cannot parse y_range = {y_range}. Min must be less than and not equal to max.")
            self.ax.set_ylim(y_range[0], y_range[1])
        elif y_range is None:
            self.ax.set_autoscaley_on(True)
        else:
            self.logger.critical("Cannot parse y_range. Unsupported type. None or tuple expected.")
            self.close_fig()
            raise TypeError(f"Cannot parse y_range = {y_range}. Only a tuple can be used.")

        if record_movie:
            self._writer_kwargs = writer_kwargs
            self._new_writer()

            if not movie_kwargs:
                movie_kwargs = {}
            if 'outfile' not in movie_kwargs:
                movie_kwargs['outfile'] = Path('glomporecording.mp4')
                self.logger.info("Saving scope recording as glomporecording.mp4")
            self._movie_kwargs = movie_kwargs

        self.logger.debug("Scope initialised successfully")

    def _new_writer(self):
        """ Constructs a new FFMpegWriter to record a movie """
        try:
            self._writer = MyFFMpegWriter(**self._writer_kwargs) if self._writer_kwargs else MyFFMpegWriter()
        except TypeError:
            warnings.warn("Unidentified key in writer_kwargs. Using default values.", UserWarning)
            self.logger.warning("Unidentified key in writer_kwargs. Using default values.")
            self._writer = MyFFMpegWriter()

    def _redraw_graph(self, force=False):
        """ Redraws the figure after new data has been added. Grabs a frame if a movie is being recorded.
            force=True overrides the normal flush counter and forces an explict reconstruction of the graph.
        """
        if self._event_counter >= self.events_per_flush or force:
            self._event_counter = 1

            # Purge old results
            if self.truncated:
                for opt_id, line in self.opt_streams.items():
                    if opt_id not in self._dead_streams:
                        done = []
                        x_vals = np.array(line.get_xdata())
                        y_vals = np.array(line.get_ydata())

                        if len(x_vals) > 0:
                            min_val = np.clip(self.x_max - self.truncated, 0, None)

                            bool_arr = x_vals >= min_val
                            x_vals = x_vals[bool_arr]
                            y_vals = y_vals[bool_arr]

                            line.set_xdata(x_vals)
                            line.set_ydata(y_vals)
                        done.append(len(x_vals) == 0)
                        if all(done):
                            self._dead_streams.add(opt_id)
                            self.logger.debug("Opt%d identified as out of scope.", opt_id)

            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.record_movie:
                if self.is_setup:
                    self.logger.debug('Grabbing frame')
                    self._writer.grab_frame()
                    self.logger.debug('Frame grabbed')
                else:
                    self.logger.error("Cannot record movie without calling setup_moviemaker first.")
                    raise RuntimeError("Cannot record movie without calling setup_moviemaker first.")
        else:
            self._event_counter += 1

    def _update_point(self, opt_id: int, track: str, pt: tuple = None):
        """ General method to add a point to a track for a specific optimizer. """

        pt_given = bool(pt)
        pt = self.get_farthest_pt(opt_id) if not pt_given else pt

        if opt_id in self._dead_streams:
            self._dead_streams.remove(opt_id)
            self.logger.warning("Receiving data for Opt%d previously identified as truncated.", opt_id)

        if pt:
            x, y = pt

            if pt_given and self.log_scale:
                y = np.log10(y)
                if np.isnan(y):
                    self.logger.error("Log10(y) returned NaN.")

            self.x_max = x if x > self.x_max else self.x_max

            if track == 'all_opt':
                line = self.opt_streams[opt_id]
            else:
                line = self.gen_streams[track]
            x_vals = np.append(line.get_xdata(), x)
            y_vals = np.append(line.get_ydata(), y)

            line.set_xdata(x_vals)
            line.set_ydata(y_vals)

    def add_stream(self, opt_id: int, opt_type: Optional[str] = None):
        """ Registers and sets up a new optimizer in the scope. """
        self.n_streams += 1

        line_style = '-'
        marker = '.'
        color = self.color_map(self.n_streams)

        self.opt_streams[opt_id] = self.ax.plot([], [], ls=line_style, marker=marker, color=color)[0]

        if opt_type:
            label = f"{opt_id}: {opt_type}"
        else:
            label = f"Optimizer {opt_id}"

        self.leg_elements.append(lines.Line2D([], [], ls=line_style, marker=marker, c=color, label=label))
        self.ax.legend(loc='upper right', handles=self.leg_elements, bbox_to_anchor=(1.35, 1))
        self.logger.debug("Added new plot set for optimizer %d", opt_id)

    def update_optimizer(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id optimizer plot."""
        x, y = pt
        if self.elitism:
            y_vals = self.opt_streams[opt_id].get_ydata()
            if len(y_vals) > 0:
                last = 10 ** y_vals[-1] if self.log_scale else y_vals[-1]
                if last < y:
                    y = last

        self._update_point(opt_id, 'all_opt', (x, y))
        self._redraw_graph()

    def update_kill(self, opt_id: int):
        """ The opt_id kill optimizer plot is updated at its final point. """
        self._update_point(opt_id, 'opt_kill')
        self._redraw_graph()

    def update_norm_terminate(self, opt_id: int):
        """ The opt_id normal optimizer plot is updated at its final point if the optimizer converged normally. """
        self._update_point(opt_id, 'opt_norm')
        self._redraw_graph()

    def update_crash_terminate(self, opt_id: int):
        """ The opt_id normal optimizer plot is updated at its final point if the optimizer crashed. """
        self._update_point(opt_id, 'opt_crash')
        self._redraw_graph()

    def update_pause(self, opt_id: int):
        """ The opt_id normal optimizer plot is updated at its final point. """
        self._update_point(opt_id, 'pause')
        self._redraw_graph()

    def update_checkpoint(self, opt_id: int):
        """ The opt_id normal optimizer plot is updated at its final point. """
        self._update_point(opt_id, 'chkpt')
        self._redraw_graph()

    def setup_moviemaker(self, path: Union[Path, str, None] = None):
        """ Setups up the movie recording framework. Must be called before the scope begins to be filled in order to
            begin generating movies correctly.

            Parameters
            ----------
            path: Optional[Path, str] = None
                An optional directory into which the movie file will be directed. Will overwrite any 'outfile' argument
                sent during scope initialisation.
        """
        if not self.record_movie:
            warnings.warn("Cannot initialise movie writer. record_movie must be True at initialisation. Aborting.",
                          UserWarning)
            self.logger.warning("Cannot initialise movie writer. record_movie must be True at initialisation. "
                                "Aborting.")
            return

        if path:
            path = (path / self._movie_kwargs['outfile']).with_suffix('.mp4')
            self._movie_kwargs['outfile'] = path

        try:
            self._writer.setup(fig=self.fig, **self._movie_kwargs)
        except TypeError:
            warnings.warn("Unidentified key in writer_kwargs. Using default values.", UserWarning)
            self.logger.warning("Unidentified key in writer_kwargs. Using default values.")
            self._writer.setup(fig=self.fig, outfile=str(self._movie_kwargs['outfile']))
        finally:
            self.is_setup = True

    def generate_movie(self):
        """ Final call to write the saved frames into a single movie. """
        if self.record_movie:
            try:
                self._redraw_graph(True)
                self._writer.finish()
            except Exception as e:
                self.logger.exception("generate_movie failed", exc_info=e)
                warnings.warn(f"Exception caught while trying to save movie: {e}", RuntimeWarning)
        else:
            self.logger.error("generate_movie called without initialisation parameter record_movie = False")
            warnings.warn("Unable to generate movie file as data was not collected during the dynamic plotting.\n"
                          "Rerun GloMPOScope with record_movie = True during initialisation.", UserWarning)

    def get_farthest_pt(self, opt_id: int) -> Tuple[float, float]:
        """ Returns the furthest evaluated point of the opt_id optimizer. """
        try:
            x = float(self.opt_streams[opt_id].get_xdata()[-1])
            y = float(self.opt_streams[opt_id].get_ydata()[-1])
        except IndexError:
            return None

        return x, y

    def close_fig(self):
        """ Matplotlib will keep a figure alive in its memory for the duration a process is alive. This can lead to
            many figures being open if GloMPO is looped in some way. The manager will explicitly call this method to
            close the matplotlib figure at the end of the optimization routine to stop figures building up in this way.
        """
        if self.record_movie and self.is_setup:
            self._writer.cleanup()
        plt.close(self.fig)

    @needs_optional_package('dill')
    def checkpoint_save(self, path: Union[Path, str] = ''):
        """ Saves the state of the scope, suitable for resumption, during a checkpoint. Path is a directory in which to
            dump the generated files.
        """
        if self.is_setup:
            self._redraw_graph(True)

        dump_variables = {}
        for var in dir(self):
            if '__' not in var and not callable(getattr(self, var)) and \
                    all([var != block for block in ('_writer', 'logger')]):
                dump_variables[var] = getattr(self, var)

        with Path(path, 'scope').open('wb') as file:
            dill.dump(dump_variables, file)

        if self.record_movie:
            warnings.warn("Movie saving is not supported with checkpointing", RuntimeWarning)
            self.logger.warning("Movie saving is not supported with checkpointing")

    @needs_optional_package('dill')
    def load_state(self, path: Union[Path, str]):
        """ Loads a saved scope state. Path is a directory containing the checkpoint files. """

        with Path(path, 'scope').open('rb') as file:
            data = dill.load(file)

        for var, val in data.items():
            setattr(self, var, val)

        if self.record_movie:
            self._new_writer()
            self.setup_moviemaker()

        if self.interactive_mode:
            self.fig.show()
