

from typing import *
from time import time
import matplotlib.animation as ani
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import warnings


# Overwrites a method in the matplotlib.animation.FFMpegWriter class which caused it to hang during movie generation

class MyFFMpegWriter(ani.FFMpegWriter):
    def cleanup(self):
        """Clean-up and collect the process used to write the movie file."""
        self._frame_sink().close()

# End Error Fix Section


class GloMPOScope:
    """ Constructs and records the dynamic plotting of optimizers run in parallel"""

    def __init__(self,
                 x_range: Tuple[float, float] = 100,
                 y_range: Tuple[float, float] = (),
                 visualise_gpr: bool = False,
                 record_movie: bool = False,
                 interactive_mode: bool = False,
                 writer_kwargs: Union[Dict[str, Any], None] = None,
                 movie_kwargs: Union[Dict[str, Any], None] = None):
        """
        Initializes the plot and movie recorder.

        Parameters
        ----------
        x_range : Union[Tuple[float, float], int, None]
            If None is provided the x-axis will automatically and continuously rescale from zero as the number of
            function evaluations increases.
            If a tuple of the form (min, max) is provided then the x-axis will be fixed to this range.
            If an integer is provided then the plot will only show the last x_range evaluations and discard earlier
            points. This is useful to make differences between optimizers visible in the late stage and also keep the
            scope operating at an adequate speed.
            Default value is set to 100.
        y_range : Tuple[float, float]
            Sets the y-axis limits of the plot, default is an empty tuple which leads the plot to automatically and
            constantly rescale the axis.
        visualise_gpr : bool
            If True the plot will show the regression itself if False only the predicted mean and uncertainty on the
            mean will be shown.
        record_movie : bool
            If True then a matplotlib.animation.FFMpegFileWriter instance is created to record the plot.
        interactive_mode : bool
            If True the plot is visible on screen during the optimization.
        writer_kwargs : Union[Dict[str, Any], None]
            Optional dictionary of arguments to be sent to the initialisation of the
            matplotlib.animation.FFMpegFileWriter class.
        movie_kwargs : Union[Dict[str, Any], None]
            Optional dictionary of arguments to be sent to matplotlib.animation.FFMpegFileWriter.setup().
        """

        plt.ion() if interactive_mode else plt.ioff()

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title("GloMPO Scope")
        self.ax.set_xlabel("Total Function Calls")
        self.ax.set_ylabel("Error")

        self.streams = {}
        self.n_streams = 0
        self.t_last = 0
        self.x_max = 0

        # Create custom legend
        self.visualise_gpr = visualise_gpr
        leg_elements = [lines.Line2D([], [], ls='-', c='black', label='Optimizer Evaluations'),
                        lines.Line2D([], [], ls='', marker='o', c='black', label='Point in Training Set'),
                        lines.Line2D([], [], ls='', marker=6, c='black', label='Hyperparam. Opt. Started'),
                        lines.Line2D([], [], ls='', marker=7, c='black', label='Hyperparam. Updated'),
                        lines.Line2D([], [], ls='', marker='x', c='black', label='Optimizer Killed'),
                        lines.Line2D([], [], ls='', marker='*', c='black', label='Optimizer Converged')]
        if visualise_gpr:
            leg_elements.append(lines.Line2D([], [], ls='-.', c='black', label='Regression'))
            leg_elements.append(lines.Line2D([], [], ls=':', c='black', label='Regression Uncertainty'))
        else:
            leg_elements.append(lines.Line2D([], [], ls='--', c='black', label='Estimated Mean'))
            leg_elements.append(patches.Patch(fc='silver', ec='black', ls=':', label='Mean Uncertainty'))

        self.ax.legend(loc='upper right', handles=leg_elements)

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
            # self.ax.set_autoscalex_on(True)
        else:
            raise TypeError(f"Cannot parse x_range = {x_range}. Only int, None_Type and tuple can be used.")

        self.ax.set_ylim(y_range[0], y_range[1]) if y_range else self.ax.set_autoscaley_on(True)

        self.record_movie = record_movie
        if record_movie:
            self.writer = MyFFMpegWriter(**writer_kwargs) if writer_kwargs else MyFFMpegWriter()
            if not movie_kwargs:
                movie_kwargs = {}
            if 'outfile' not in movie_kwargs:
                movie_kwargs['outfile'] = 'glomporecording.mp4'
            self.writer.setup(fig=self.fig, **movie_kwargs)
            os.makedirs("_tmp_movie_grabs", exist_ok=True)

    def _redraw_graph(self):
        if time() - self.t_last > 1:
            self.t_last = time()

            # Purge old results
            if self.truncated:
                for axes in self.streams.values():
                    for line in axes.values():
                        if not isinstance(line, patches.Rectangle):
                            x_vals = np.array(line.get_xdata())
                            y_vals = np.array(line.get_ydata())

                            if len(x_vals) > 0:
                                min_val = self.x_max - self.truncated
                                x_vals = x_vals[x_vals >= min_val]
                                y_vals = y_vals[-len(x_vals):]

                                line.set_xdata(x_vals)
                                line.set_ydata(y_vals)
                        else:
                            line.xy = (self.x_max - self.truncated, line.xy[1])

            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if self.record_movie:
                os.chdir("_tmp_movie_grabs")
                self.writer.grab_frame()
                os.chdir("..")

    def _update_point(self, opt_id: int, track: str, pt: tuple = None):

        if not pt:
            pt = self.get_farthest_pt(opt_id)

        self.x_max = pt[0] if pt[0] > self.x_max else self.x_max

        line = self.streams[opt_id][track]
        x_vals = np.append(line.get_xdata(), pt[0])
        y_vals = np.append(line.get_ydata(), pt[1])

        if self.truncated:
            min_val = np.max(x_vals) - self.truncated
            x_vals = x_vals[x_vals >= min_val]
            y_vals = y_vals[-len(x_vals):]

        line.set_xdata(x_vals)
        line.set_ydata(y_vals)

    def add_stream(self, opt_id):
        self.n_streams += 1
        self.streams[opt_id] = {'all_opt': self.ax.plot([], [])[0],  # Follows every optimizer iteration
                                'train_pts': self.ax.plot([], [], ls='', marker='o')[0],  # Points in training set
                                'hyper_init': self.ax.plot([], [], ls='', marker=6)[0],  # Hyperparms job start
                                'hyper_up': self.ax.plot([], [], ls='', marker=7)[0],  # Hyperparms changed
                                'opt_kill': self.ax.plot([], [], ls='', marker='x', zorder=500)[0],  # Killed opt
                                'opt_norm': self.ax.plot([], [], ls='', marker='*', zorder=500)[0]}  # Converged opt
        if self.visualise_gpr:
            self.streams[opt_id]['gpr_mean'] = self.ax.plot([], [], ls='-.')[0]  # GPR
            self.streams[opt_id]['gpr_upper'] = self.ax.plot([], [], ls=':')[0]  # GPR Upper Sigma
            self.streams[opt_id]['gpr_lower'] = self.ax.plot([], [], ls=':')[0]  # GPR Lower Sigma
        else:
            self.streams[opt_id]['mean'] = self.ax.plot([], ls='--')[0]  # Plots the mean functions
            self.streams[opt_id]['st_dev'] = patches.Rectangle((0, 0), 0, 0, ls=':')  # Plots the uncertainty on mean
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

    def update_optimizer(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id optimizer plot."""
        self._update_point(opt_id, 'all_opt', pt)

    def update_scatter(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id training data plot."""
        self._update_point(opt_id, 'train_pts', pt)
        self._redraw_graph()

    def update_mean(self, opt_id: int, mu: float, sigma: float):
        """ Given mu and sigma is used to update the opt_id mean and uncertainty plots."""
        # Mean line
        line = self.streams[opt_id]['mean']
        x_max = self.get_farthest_pt(opt_id)[0]
        self.x_max = x_max if x_max > self.x_max else self.x_max
        x_min = np.clip(x_max - self.truncated, 0, None) if self.truncated else 0
        line.set_xdata((x_min, x_max))
        line.set_ydata((mu, mu))

        # Uncertainty Rectangle
        rec = self.streams[opt_id]['st_dev']
        rec.xy = (x_min, mu - 2 * sigma)
        width = self.truncated if self.truncated else x_max
        rec.set_width(width)
        rec.set_height(4 * sigma)
        self._redraw_graph()

    def update_opt_start(self, opt_id: int):
        """ Given pt tuple is used to update the opt_id start hyperparameter optimizer plot."""
        self._update_point(opt_id, 'hyper_init')
        self._redraw_graph()

    def update_opt_end(self, opt_id: int):
        """ Given pt tuple is used to update the opt_id end hyperparameter optimizer plot."""
        self._update_point(opt_id, 'hyper_up')
        self._redraw_graph()

    def update_kill(self, opt_id: int):
        """ The opt_id kill optimizer plot is updated at its final point. """
        self._update_point(opt_id, 'opt_kill')
        self._redraw_graph()

    def update_norm_terminate(self, opt_id: int):
        """ The opt_id normal optimizer plot is updated at its final point. """
        self._update_point(opt_id, 'opt_norm')
        self._redraw_graph()

    def update_gpr(self, opt_id: int, x: np.ndarray, y: np.ndarray, lower_sig: np.ndarray, upper_sig: np.ndarray):
        """ Given mu and sigma is used to update the opt_id mean and uncertainty plots."""
        # Mean line
        line = self.streams[opt_id]['gpr_mean']

        if self.truncated:
            min_val = np.max(x) - self.truncated
            x = x[x >= min_val]
            y = y[-len(x):]
            lower_sig = lower_sig[-len(x):]
            upper_sig = upper_sig[-len(x):]

        line.set_xdata(x)
        line.set_ydata(y)

        # Uncertainty
        line = self.streams[opt_id]['gpr_lower']
        line.set_xdata(x)
        line.set_ydata(lower_sig)
        line = self.streams[opt_id]['gpr_upper']
        line.set_xdata(x)
        line.set_ydata(upper_sig)
        self._redraw_graph()

    def generate_movie(self):
        """ Final call to write the saved frames into a single movie. """
        if self.record_movie:
            try:
                os.chdir("_tmp_movie_grabs")
                self.writer.finish()
                files = [file for file in os.listdir(".") if ".mp4" in file]
                for file in files:
                    shutil.move(file, f"../{file}")
                os.chdir("..")
            except Exception as e:
                warnings.warn(f"Exception caught while trying to save movie: {e}", UserWarning)
            finally:
                shutil.rmtree("_tmp_movie_grabs", ignore_errors=True)
        else:
            warnings.warn("Unable to generate movie file as data was not collected during the dynamic plotting.\n"
                          "Rerun GloMPOScope with record_movie = True during initialisation.", UserWarning)

    def get_farthest_pt(self, opt_id: int):
        """ Returns the furthest evaluated point of the opt_id optimizer. """
        x = float(self.streams[opt_id]['all_opt'].get_xdata()[-1])
        y = float(self.streams[opt_id]['all_opt'].get_ydata()[-1])

        return x, y
