

from typing import *
import matplotlib.animation as ani
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class ParallelOptimizerScope:
    """ Constructs and records the dynamic plotting of optimizers run in parallel"""

    def __init__(self, num_streams: int,
                 x_range: Tuple[float, float] = (),
                 y_range: Tuple[float, float] = (),
                 visualise_gpr: bool = False,
                 record_movie: bool = False,
                 writer_kwargs: Union[Dict[str, Any], None] = None,
                 movie_kwargs: Union[Dict[str, Any], None] = None):
        """
        Initializes the plot and movie recorder.

        Parameters
        ----------
        num_streams : int
            Total number of optimizers being run in parallel.
        x_range : Tuple[float, float]
            Sets the x-axis limits of the plot, default is an empty tuple which leads the plot to automatically and
            constantly rescale the axis.
        y_range : Tuple[float, float]
            Sets the y-axis limits of the plot, default is an empty tuple which leads the plot to automatically and
            constantly rescale the axis.
        visualise_gpr : bool
            If True the plot will show the regression itself if False only the predicted mean and uncertainty on the
            mean will be shown.
        record_movie : bool
            If True then a matplotlib.animation.FFMpegFileWriter instance is created to record the plot.
        writer_kwargs : Union[Dict[str, Any], None]
            Optional dictionary of arguments to be sent to the initialisation of the
            matplotlib.animation.FFMpegFileWriter class.
        movie_kwargs : Union[Dict[str, Any], None]
            Optional dictionary of arguments to be sent to matplotlib.animation.FFMpegFileWriter.setup().
        """
        # TODO Set workdir
        # TODO delete all tmp pics if the program crashes
        # TODO NB DO NOT INITIALISE WITH STREAMS. CREATE ADD_STREAMS METHOD
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Error")

        self.streams = {}
        self.n_streams = 0

        # Create custom legend
        self.visualise_gpr = visualise_gpr
        leg_elements = [lines.Line2D([], [], ls='-', c='black', label='Optimizer Evaluations'),
                        lines.Line2D([], [], ls='', marker='o', c='black', label='Point in Training Set'),
                        lines.Line2D([], [], ls='', marker=6, c='black', label='Hyperparam. Opt. Started'),
                        lines.Line2D([], [], ls='', marker=7, c='black', label='Hyperparam. Updated'),
                        lines.Line2D([], [], ls='', marker='x', c='black', label='Optimizer Killed'),
                        lines.Line2D([], [], ls='', marker='*', c='black', label='Optimizer Converged')]
        if visualise_gpr:
            leg_elements.append(lines.Line2D([], [], ls='-.', c='black', label='Regression and Uncertainty'))
        else:
            leg_elements.append(lines.Line2D([], [], ls='--', c='black', label='Estimated Mean'))
            leg_elements.append(patches.Patch(fc='silver', ec='black', ls=':', label='Mean Uncertainty'))

        self.ax.legend(loc='upper right', handles=leg_elements)

        self.ax.set_xlim(x_range[0], x_range[1]) if x_range else self.ax.set_autoscalex_on(True)
        self.ax.set_ylim(y_range[0], y_range[1]) if y_range else self.ax.set_autoscaley_on(True)

        self.record_movie = record_movie
        if record_movie:
            self.writer = ani.FFMpegFileWriter(**writer_kwargs) if writer_kwargs else ani.FFMpegFileWriter()
            if not movie_kwargs:
                movie_kwargs = {}
            if 'outfile' not in movie_kwargs:
                movie_kwargs['outfile'] = 'glomporecording.mp4'
            self.writer.setup(fig=self.fig, **movie_kwargs)

    def _update(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.record_movie:
            self.writer.grab_frame()

    def add_stream(self, opt_id):
        self.n_streams += 1
        self.streams[opt_id] = {'all_opt': self.ax.plot([], [])[0],  # 0 Follows every optimizer iteration
                                'train_pts': self.ax.plot([], [], ls='', marker='o')[0],  # 1 Points in training set
                                'mean': self.ax.plot([], ls='--')[0],  # 2 Plots the mean functions
                                'st_dev': patches.Rectangle((0, 0), 0, 0, ls=':'),  # 3 Plots the uncertainty on mean
                                'hyper_init': self.ax.plot([], [], ls='', marker=6)[0],  # 4 Hyperparms job start
                                'hyper_up': self.ax.plot([], [], ls='', marker=7)[0],  # 5 Hyperparms changed
                                'opt_kill': self.ax.plot([], [], ls='', marker='x', zorder=500)[0],  # 6 Killed opt
                                'opt_norm': self.ax.plot([], [], ls='', marker='*', zorder=500)[0],  # 7 Converged opt
                                'gpr_mean': self.ax.plot([], [], ls='-.')[0],  # 8 GPR
                                'gpr_upper': self.ax.plot([], [], ls='-.')[0],  # 9 GPR Upper Sigma
                                'gpr_lower': self.ax.plot([], [], ls='-.')[0]}  # 10 GPR Lower Sigma

        # Match colors for a single optimisation
        colors = plt.get_cmap("tab10")
        for line in self.streams[opt_id]:
            color = colors(self.n_streams)
            if any([line == _ for _ in ['train_pts', 'hyper_init', 'hyper_up']]):
                color = tuple([0.75, 0.75, 0.75, 1] * np.array(color))
            elif line == 'st_dev':
                color = tuple([1, 1, 1, 0.5] * np.array(color))
            elif any([line == _ for _ in ['opt_kill', 'opt_norm']]):
                color = 'red'
            self.streams[opt_id][line].set_color(color)
        self.ax.add_patch(self.streams[opt_id]['st_dev'])

    def update_optimizer(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id optimizer plot."""
        line = self.streams[opt_id]['all_opt']
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_scatter(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id training data plot."""
        line = self.streams[opt_id]['train_pts']
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_mean(self, opt_id: int, mu: float, sigma: float):
        """ Given mu and sigma is used to update the opt_id mean and uncertainty plots."""
        # Mean line
        line = self.streams[opt_id]['mean']
        line.set_xdata((0, self.ax.get_xlim()[1]))
        line.set_ydata((mu, mu))

        # Uncertainty Rectangle
        rec = self.streams[opt_id]['st_dev']
        rec.xy = (0, mu - 2 * sigma)
        rec.set_width(self.ax.get_xlim()[1])
        rec.set_height(4 * sigma)
        self._update()

    def update_opt_start(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id start hyperparameter optimizer plot."""
        line = self.streams[opt_id]['hyper_init']
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_opt_end(self, opt_id: int, pt: tuple):
        """ Given pt tuple is used to update the opt_id end hyperparameter optimizer plot."""
        line = self.streams[opt_id]['hyper_up']
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_kill(self, opt_id: int):
        """ The opt_id kill optimizer plot is updated at its final point. """
        # Add dead optimizer marker
        line = self.streams[opt_id]['opt_kill']
        x_pt, y_pt = self.get_farthest_pt(opt_id)
        line.set_xdata(x_pt)
        line.set_ydata(y_pt)

        # Shrink the uncertainty patch
        rec = self.streams[opt_id]['opt_kill']
        rec.set_width(y_pt)

        self._update()

    def update_norm_terminate(self, opt_id: int):
        """ The opt_id normal optimizer plot is updated at its final point. """
        # Add dead optimizer marker
        line = self.streams[opt_id]['opt_norm']
        x_pt, y_pt = self.get_farthest_pt(opt_id)
        line.set_xdata(x_pt)
        line.set_ydata(y_pt)

        # Shrink the uncertainty patch
        rec = self.streams[opt_id]['opt_norm']
        rec.set_width(y_pt)

        self._update()

    def update_gpr(self, opt_id: int, x: np.ndarray, y: np.ndarray, lower_sig: np.ndarray, upper_sig: np.ndarray):
        """ Given mu and sigma is used to update the opt_id mean and uncertainty plots."""
        # Mean line
        line = self.streams[opt_id]['gpr']
        line.set_xdata(x)
        line.set_ydata(y)

        # Uncertainty
        line = self.streams[opt_id]['gpr_lower']
        line.set_xdata(x)
        line.set_ydata(lower_sig)
        line = self.streams[opt_id]['gpr_upper']
        line.set_xdata(x)
        line.set_ydata(upper_sig)
        self._update()

    def generate_movie(self):
        """ Final call to write the saved frames into a single movie. """
        if self.record_movie:
            self.writer.finish()
        else:
            print("Unable to generate movie file as data was not collected during the dynamic plotting.\n"
                  "Rerun ParallelOptimizerScope with record_movie = True during initialisation.")

    def get_farthest_pt(self, opt_id: int):
        """ Returns the furthest evaluated point of the 'n_stream'th optimizer. """
        x_pt_all = float(self.streams[opt_id]['all_opt'].get_xdata()[-1])
        y_pt_all = float(self.streams[opt_id]['all_opt'].get_ydata()[-1])

        x_pt_tps = float(self.streams[opt_id]['all_opt'].get_xdata()[-1])
        y_pt_tps = float(self.streams[opt_id]['all_opt'].get_ydata()[-1])

        if x_pt_all > x_pt_tps:
            x, y = x_pt_all, y_pt_all
        else:
            x, y = x_pt_tps, y_pt_tps

        return x, y
