

""" Contains the GloMPOScope class which is a useful extension allowing a user to visualize GloMPO's behaviour. """


from typing import *
from time import time
import warnings

import matplotlib.animation as ani
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from ..common.wrappers import catch_user_interrupt, decorate_all_methods


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
    """ Constructs and records the dynamic plotting of optimizers run in parallel"""

    def __init__(self,
                 x_range: Union[Tuple[float, float], int, None] = 300,
                 y_range: Optional[Tuple[float, float]] = None,
                 log_scale: bool = False,
                 record_movie: bool = False,
                 interactive_mode: bool = False,
                 writer_kwargs: Optional[Dict[str, Any]] = None,
                 movie_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initializes the plot and movie recorder.

        Parameters
        ----------
        x_range : Union[Tuple[float, float], int, None] = 300
            If None is provided the x-axis will automatically and continuously rescale from zero as the number of
            function evaluations increases.
            If a tuple of the form (min, max) is provided then the x-axis will be fixed to this range.
            If an integer is provided then the plot will only show the last x_range evaluations and discard earlier
            points. This is useful to make differences between optimizers visible in the late stage and also keep the
            scope operating at an adequate speed.
        y_range : Optional[Tuple[float, float]] = None
            Sets the y-axis limits of the plot, default is an empty tuple which leads the plot to automatically and
            constantly rescale the axis.
        log_scale: bool = False
            If True, the base 10 logarithm of y values are displayed on the scope. This can be used in conjunction
            with the y_range option and will be interpretted in the log-scale.
        record_movie : bool = False
            If True then a matplotlib.animation.FFMpegWriter instance is created to record the plot.
        interactive_mode : bool = False
            If True the plot is visible on screen during the optimization.
        writer_kwargs : Optional[Dict[str, Any]] = None
            Optional dictionary of arguments to be sent to the initialisation of the matplotlib.animation.FFMpegWriter
            class.
        movie_kwargs : Optional[Dict[str, Any]] = None
            Optional dictionary of arguments to be sent to matplotlib.animation.FFMpegWriter.setup().
        """

        self.streams = {}
        self.dead_streams = set()
        self.n_streams = 0
        self.t_last = 0
        self.x_max = 0
        self.log_scale = log_scale

        plt.ion() if interactive_mode else plt.ioff()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title("GloMPO Scope")
        self.ax.set_xlabel("Total Function Calls")
        if self.log_scale:
            self.ax.set_ylabel("Log10(Error)")
        else:
            self.ax.set_ylabel("Error")

        # Create custom legend
        self.leg_elements = [lines.Line2D([], [], ls='-', c='black', label='Optimizer Evaluations'),
                             lines.Line2D([], [], ls='', marker='o', c='black', label='Point in Training Set'),
                             lines.Line2D([], [], ls='', marker=6, c='black', label='Hunt Started'),
                             lines.Line2D([], [], ls='', marker=7, c='black', label='Hunt Result'),
                             lines.Line2D([], [], ls='', marker='x', c='black', label='Optimizer Killed'),
                             lines.Line2D([], [], ls='', marker='*', c='black', label='Optimizer Converged'),
                             lines.Line2D([], [], ls='-.', c='black', label='Asymptote'),
                             lines.Line2D([], [], ls=':', c='black', label='Asymptote Uncertainty')]

        self.ax.legend(loc='upper right', handles=self.leg_elements, bbox_to_anchor=(1.35, 1))

        # Setup and shrink axis position to fit legend
        box = self.ax.get_position()
        self.ax.set_position([0.85 * box.x0, box.y0, 0.85 * box.width, box.height])

        # Setup axis limits
        self.truncated = None
        if x_range is None:
            self.ax.set_autoscalex_on(True)
        elif isinstance(x_range, tuple):
            if x_range[0] >= x_range[1]:
                raise ValueError(f"Cannot parse x_range = {x_range}. Min must be less than and not equal to max.")
            self.ax.set_xlim(x_range[0], x_range[1])
        elif isinstance(x_range, int):
            if x_range < 2:
                raise ValueError(f"Cannot parse x_range = {x_range}. Value larger than 1 required.")
            self.truncated = x_range
        else:
            raise TypeError(f"Cannot parse x_range = {x_range}. Only int, None_Type and tuple can be used.")

        if isinstance(y_range, tuple):
            if y_range[0] >= y_range[1]:
                raise ValueError(f"Cannot parse y_range = {y_range}. Min must be less than and not equal to max.")
            self.ax.set_ylim(y_range[0], y_range[1])
        elif y_range is None:
            self.ax.set_autoscaley_on(True)
        else:
            raise TypeError(f"Cannot parse y_range = {y_range}. Only tuple can be used.")

        self.record_movie = record_movie
        self.movie_name = None
        if record_movie:
            try:
                self.writer = MyFFMpegWriter(**writer_kwargs) if writer_kwargs else MyFFMpegWriter()
            except TypeError:
                warnings.warn("Unidentified key in writer_kwargs. Using default values.", UserWarning)
                self.writer = MyFFMpegWriter()
            if not movie_kwargs:
                movie_kwargs = {}
            if 'outfile' not in movie_kwargs:
                movie_kwargs['outfile'] = 'glomporecording.mp4'
                self.movie_name = 'glomporecording.mp4'
            else:
                self.movie_name = movie_kwargs['outfile']
            try:
                self.writer.setup(fig=self.fig, **movie_kwargs)
            except TypeError:
                warnings.warn("Unidentified key in writer_kwargs. Using default values.", UserWarning)
                self.writer.setup(fig=self.fig, outfile='glomporecording.mp4')

    def _redraw_graph(self):
        """ Redraws the figure after new data has been added. Grabs a frame if a movie is being recorded. """
        if time() - self.t_last > 1:
            self.t_last = time()

            # Purge old results
            if self.truncated:
                for opt_id in self.streams:
                    if opt_id not in self.dead_streams:
                        done = []
                        for line in self.streams[opt_id].values():
                            if not isinstance(line, patches.Rectangle):
                                x_vals = np.array(line.get_xdata())
                                y_vals = np.array(line.get_ydata())

                                if len(x_vals) > 0:
                                    min_val = np.clip(self.x_max - self.truncated, 0, None)
                                    if line is not self.streams[opt_id]['mean']:
                                        bool_arr = x_vals >= min_val
                                        x_vals = x_vals[bool_arr]
                                        y_vals = y_vals[bool_arr]

                                        line.set_xdata(x_vals)
                                        line.set_ydata(y_vals)
                                    else:
                                        line.set_xdata((min_val, self.x_max))
                                done.append(True) if len(x_vals) == 0 else done.append(False)
                            else:
                                line.xy = (np.clip(self.x_max - self.truncated, 0, None), line.xy[1])
                        if all(done):
                            self.dead_streams.add(opt_id)

            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.record_movie:
                self.writer.grab_frame()

    def _update_point(self, opt_id: int, track: str, pt: tuple = None):
        """ General method to add a point to a track for a specific optimizer. """

        pt_given = bool(pt)
        pt = self.get_farthest_pt(opt_id) if not pt_given else pt

        if opt_id in self.dead_streams:
            self.dead_streams.remove(opt_id)

        if pt:
            x, y = pt

            if pt_given and self.log_scale:
                y = np.log10(y)

            self.x_max = x if x > self.x_max else self.x_max

            line = self.streams[opt_id][track]
            x_vals = np.append(line.get_xdata(), x)
            y_vals = np.append(line.get_ydata(), y)

            line.set_xdata(x_vals)
            line.set_ydata(y_vals)

    def add_stream(self, opt_id):
        """ Registers and sets up a new optimizer in the scope. """

        self.n_streams += 1
        self.streams[opt_id] = {'all_opt': self.ax.plot([], [])[0],  # Follows every optimizer iteration
                                'train_pts': self.ax.plot([], [], ls='', marker='o')[0],  # Points in training set
                                'hunt_init': self.ax.plot([], [], ls='', marker=6)[0],  # Hunt start
                                'hunt_up': self.ax.plot([], [], ls='', marker=7)[0],  # Hunt result
                                'opt_kill': self.ax.plot([], [], ls='', marker='x', zorder=500)[0],  # Killed opt
                                'opt_norm': self.ax.plot([], [], ls='', marker='*', zorder=500)[0],  # Converged opt
                                'mean': self.ax.plot([], ls='--')[0],  # Plots the mean functions
                                'st_dev': patches.Rectangle((0, 0), 0, 0, ls=':')}  # Plots the uncertainty on mean
        self.ax.add_patch(self.streams[opt_id]['st_dev'])

        # Match colors for a single optimisation
        if self.n_streams < 20:
            colors = plt.get_cmap("tab20")
            threshold = 0
        elif self.n_streams < 40:
            colors = plt.get_cmap("tab20b")
            threshold = 20
        elif self.n_streams < 60:
            colors = plt.get_cmap("tab20c")
            threshold = 40
        elif self.n_streams < 69:
            colors = plt.get_cmap("Set1")
            threshold = 60
        elif self.n_streams < 77:
            colors = plt.get_cmap("Set2")
            threshold = 69
        elif self.n_streams < 89:
            colors = plt.get_cmap("Set3")
            threshold = 77
        else:
            colors = plt.get_cmap("Dark2")
            threshold = 89

        for line in self.streams[opt_id]:
            color = colors(self.n_streams - threshold)
            if any([line == _ for _ in ['train_pts', 'hyper_init', 'hyper_up']]):
                color = tuple([0.75, 0.75, 0.75, 1] * np.array(color))
            elif line == 'st_dev':
                color = tuple([1, 1, 1, 0.5] * np.array(color))
            elif any([line == _ for _ in ['opt_kill', 'opt_norm']]):
                color = 'red'
            self.streams[opt_id][line].set_color(color)
        self.leg_elements.append(lines.Line2D([], [], ls='-', c=colors(self.n_streams - threshold),
                                              label=f'Optimizer {opt_id}'))
        self.ax.legend(loc='upper right', handles=self.leg_elements, bbox_to_anchor=(1.35, 1))

    def update_optimizer(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id optimizer plot."""
        self._update_point(opt_id, 'all_opt', pt)
        self._redraw_graph()

    def update_scatter(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id training data plot."""
        self._update_point(opt_id, 'train_pts', pt)
        self._redraw_graph()

    def update_mean(self, opt_id: int, median: float, lower: float, upper: float):
        """ Given median, lower and upper confidence levels are used to update the opt_id asymptote and uncertainty
            plots.
        """

        if self.log_scale:
            median = np.log10(median)
            lower = np.log10(lower)
            upper = np.log10(upper)

        # Mean line
        line = self.streams[opt_id]['mean']
        x_min = np.clip(self.x_max - self.truncated, 0, None) if self.truncated else 0
        line.set_xdata((x_min, self.x_max))
        line.set_ydata((median, median))

        # Uncertainty Rectangle
        rec = self.streams[opt_id]['st_dev']
        rec.xy = (x_min, lower)
        width = self.truncated if self.truncated else self.x_max
        rec.set_width(width)
        rec.set_height(upper - lower)
        self._redraw_graph()

    def update_hunt_start(self, opt_id: int):
        """ Given pt tuple is used to update the opt_id start hunt plot."""
        self._update_point(opt_id, 'hunt_init')
        self._redraw_graph()

    def update_hunt_end(self, opt_id: int):
        """ Given pt tuple is used to update the opt_id end hunt plot."""
        self._update_point(opt_id, 'hunt_up')
        self._redraw_graph()

    def update_kill(self, opt_id: int):
        """ The opt_id kill optimizer plot is updated at its final point. """
        self._update_point(opt_id, 'opt_kill')

        rec = self.streams[opt_id]['st_dev']
        rec.set_width(0)
        rec.set_height(0)

        line = self.streams[opt_id]['mean']
        line.set_xdata([])
        line.set_ydata([])

        self._redraw_graph()

    def update_norm_terminate(self, opt_id: int):
        """ The opt_id normal optimizer plot is updated at its final point. """
        self._update_point(opt_id, 'opt_norm')

        rec = self.streams[opt_id]['st_dev']
        rec.set_width(0)
        rec.set_height(0)

        line = self.streams[opt_id]['mean']
        line.set_xdata([])
        line.set_ydata([])

        self._redraw_graph()

    def generate_movie(self):
        """ Final call to write the saved frames into a single movie. """
        if self.record_movie:
            try:
                self.writer.finish()
            except Exception as e:
                warnings.warn(f"Exception caught while trying to save movie: {e}", RuntimeWarning)
        else:
            warnings.warn("Unable to generate movie file as data was not collected during the dynamic plotting.\n"
                          "Rerun GloMPOScope with record_movie = True during initialisation.", UserWarning)

    def get_farthest_pt(self, opt_id: int) -> Tuple[float, float]:
        """ Returns the furthest evaluated point of the opt_id optimizer. """
        try:
            x = float(self.streams[opt_id]['all_opt'].get_xdata()[-1])
            y = float(self.streams[opt_id]['all_opt'].get_ydata()[-1])
        except IndexError:
            return None

        return x, y
