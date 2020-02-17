

from typing import Union
import matplotlib.animation as ani
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class ParallelOptimizerScope:
    """ Constructs and records the dynamic plotting of optimizers run in parallel"""

    def __init__(self, num_streams: int, x_range: tuple = (), y_range: tuple = (), visualise: int = 0,
                 record_movie: bool = False, writer_args: Union[dict, None] = None,
                 movie_args: Union[dict, None] = None):
        """
        Initializes the plot and movie recorder.

        Parameters
        ----------
        num_streams : int
            Total number of optimizers being run in parallel.
        x_range : tuple
            Sets the x-axis limits of the plot, default is an empty tuple which leads the plot to automatically and
            constantly rescale the axis.
        y_range : tuple
            Sets the y-axis limits of the plot, default is an empty tuple which leads the plot to automatically and
            constantly rescale the axis.
        visualise : int
            Accepts values 0 and 1. The plot will show the regression itself if set to 0 and will show only the
            predicted mean and uncertainty if set to 1.
        record_movie : bool
            If True then a matplotlib.animation.FFMpegWriter instance is created to record the plot.
        writer_args : Union[dict, None]
            Optional dictionary of arguments to be sent to the initialisation of the matplotlib.animation.FFMpegWriter
            class.
        movie_args : Union[dict, None]
            Optional dictionary of arguments to be sent to matplotlib.animation.FFMpegWriter.setup().
        """
        # TODO Set workdir
        # TODO delete all tmp pics if the program crashes
        # TODO use the visualise variable
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.streams = [[self.ax.plot([], [])[0],  # 0 Follows every optimizer iteration
                         self.ax.plot([], [], ls='', marker='o')[0],  # 1 Plots the points added to the train sets
                         self.ax.plot([], ls='--')[0],  # 2 Plots the mean functions
                         patches.Rectangle((0, 0), 0, 0, ls=':'),  # 3 Plots the uncertainty on the mean
                         self.ax.plot([], [], ls='', marker=6)[0],  # 4 Plots where optimisation jobs are started
                         self.ax.plot([], [], ls='', marker=7)[0],  # 5 Plots where hyperparms are changed
                         self.ax.plot([], [], ls='', marker='x', zorder=500)[0],  # 6 Plots termination points
                         self.ax.plot([], [], ls='-.')[0],  # 7 GPR
                         self.ax.plot([], [], ls='-.')[0],  # 8 GPR Upper Sigma
                         self.ax.plot([], [], ls='-.')[0]]  # 9 GPR Lower Sigma
                        for _ in range(num_streams)]

        # Match colors for a single optimisation
        for i, stream in enumerate(self.streams):
            colors = plt.get_cmap("tab10")
            for j in range(len(stream)):
                color = colors(i)
                if any([j == _ for _ in [1, 4, 5]]):
                    color = tuple([0.75, 0.75, 0.75, 1] * np.array(color))
                elif j == 3:
                    color = tuple([1, 1, 1, 0.5] * np.array(color))
                elif j == 6:
                    color = 'red'
                stream[j].set_color(color)
        for stream in self.streams:
            self.ax.add_patch(stream[3])

        # Create custom legend
        leg_elements = [lines.Line2D([], [], ls='-', c='black', label='Optimizer Evaluations'),
                        lines.Line2D([], [], ls='', marker='o', c='black', label='Point in Training Set'),
                        lines.Line2D([], [], ls='--', c='black', label='Estimated Mean'),
                        patches.Patch(fc='silver', ec='black', ls=':', label='Mean Uncertainty'),
                        lines.Line2D([], [], ls='', marker=6, c='black', label='Hyperparam. Opt. Started'),
                        lines.Line2D([], [], ls='', marker=7, c='black', label='Hyperparam. Updated'),
                        lines.Line2D([], [], ls='', marker='x', c='black', label='Optimizer Killed'),
                        lines.Line2D([], [], ls='-.', c='black', label='Regression and Uncertainty')]
        self.ax.legend(loc='upper right', handles=leg_elements)

        self.ax.set_xlim(x_range[0], x_range[1]) if x_range else self.ax.set_autoscalex_on(True)
        self.ax.set_ylim(y_range[0], y_range[1]) if y_range else self.ax.set_autoscaley_on(True)

        self.record_movie = record_movie
        if record_movie:
            self.writer = ani.FFMpegFileWriter(**writer_args)
            self.writer.setup(fig=self.fig, **movie_args)

    def _update(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.record_movie:
            self.writer.grab_frame()

    def update_optimizer(self, n_stream: int, pt: tuple):
        """ Given pt tuple is used to update the 'n_stream'th optimizer plot."""
        line = self.streams[n_stream][0]
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_scatter(self, n_stream: int, pt: tuple):
        """ Given pt tuple is used to update the 'n_stream'th training data plot."""
        line = self.streams[n_stream][1]
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_mean(self, n_stream: int, mu: float, sigma: float):
        """ Given mu and sigma is used to update the 'n_stream'th mean and uncertainty plots."""
        # Mean line
        line = self.streams[n_stream][2]
        line.set_xdata((0, self.ax.get_xlim()[1]))
        line.set_ydata((mu, mu))

        # Uncertainty Rectangle
        rec = self.streams[n_stream][3]
        rec.xy = (0, mu - 2 * sigma)
        rec.set_width(self.ax.get_xlim()[1])
        rec.set_height(4 * sigma)
        self._update()

    def update_opt_start(self, n_stream: int, pt: tuple):
        """ Given pt tuple is used to update the 'n_stream'th start hyperparameter optimizer plot."""
        line = self.streams[n_stream][4]
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_opt_end(self, n_stream: int, pt: tuple):
        """ Given pt tuple is used to update the 'n_stream'th end hyperparameter optimizer plot."""
        line = self.streams[n_stream][5]
        line.set_xdata(np.append(line.get_xdata(), pt[0]))
        line.set_ydata(np.append(line.get_ydata(), pt[1]))
        self._update()

    def update_kill(self, n_stream: int):
        """ The 'n_stream'th kill optimizer plot is updated at its final point. """
        # Add dead optimizer marker
        line = self.streams[n_stream][6]
        x_pt, y_pt = self.get_farthest_pt(n_stream)
        line.set_xdata(x_pt)
        line.set_ydata(y_pt)

        # Shrink the uncertainty patch
        rec = self.streams[n_stream][3]
        rec.set_width(y_pt)

        self._update()

    def update_gpr(self, n_stream: int, x: np.ndarray, y: np.ndarray, lower_sig: np.ndarray, upper_sig: np.ndarray):
        """ Given mu and sigma is used to update the 'n_stream'th mean and uncertainty plots."""
        # Mean line
        line = self.streams[n_stream][7]
        line.set_xdata(x)
        line.set_ydata(y)

        # Uncertainty
        line = self.streams[n_stream][8]
        line.set_xdata(x)
        line.set_ydata(lower_sig)
        line = self.streams[n_stream][9]
        line.set_xdata(x)
        line.set_ydata(upper_sig)
        self._update()

    def generate_movie(self):
        """ Final call to write the saved frames into a single movie. """
        if self.record_movie:
            self.writer.finish()
        else:
            print("Unable to generate movie file as data was not collected during the dynamic plotting.\n"
                  "Rerun DynamicPlot with record_movie = True during initialisation.")

    def get_farthest_pt(self, n_stream: int):
        """ Returns the furthest evaluated point of the 'n_stream'th optimizer. """
        x_pt = float(self.streams[n_stream][0].get_xdata()[-1])
        y_pt = float(self.streams[n_stream][0].get_ydata()[-1])
        return x_pt, y_pt
