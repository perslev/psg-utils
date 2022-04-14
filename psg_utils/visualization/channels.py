import logging
import numpy as np
import matplotlib.pyplot as plt
from mne.channels import read_layout
from psg_utils import defaults

logger = logging.getLogger(__name__)


class SensorPositions(object):
    def __init__(self,
                 layout='biosemi',
                 keep_channels=None,
                 drop_channels=("Iz", "P9", "P10")):
        _layout = read_layout(layout)
        names, pos = np.array(_layout.names), np.array(_layout.pos)
        drop_channels = drop_channels or []
        if np.in1d(drop_channels, keep_channels or []).any():
            raise ValueError("Keep- and drop channel lists overlap.")
        keep_channels = keep_channels or names

        # Mask out keep and drop channels
        keep_mask = np.in1d(names, keep_channels)
        drop_mask = np.in1d(names, drop_channels)
        mask = keep_mask & (~drop_mask)
        names, pos = names[mask], pos[mask]
        self.names = names
        assert np.all(np.isin(names, keep_channels))
        pos = np.array([(p[0] + p[2] / 2., p[1] + p[3] / 2.) for p in pos])
        self.pos = pos - np.mean(pos, axis=0)

        # order according to keep_channels
        order = [keep_channels.index(n) for n in names]
        self.names = self.names[order]
        self.pos = self.pos[order]


class HeadFigure(object):
    def __init__(self,
                 sensors=None,
                 fig_size=None,
                 head_outline=True,
                 legend=True,
                 linewidth=2.5):
        s = (fig_size or max(plt.rcParams['figure.figsize']),) * 2
        self.fig = plt.figure(figsize=s)
        self.fig.subplots_adjust(left=0, bottom=0.15, right=1, top=0.85)
        self.ax = self.fig.add_subplot(111)
        self.head_outline = head_outline
        self.lw = linewidth
        self.cr = 0.025
        self.circles_ = None
        self.title_ = None
        if sensors is not None:
            self.set_sensors(sensors)
        if legend:
            self.set_legend()

    def reset(self):
        self.ax.cla()
        self.ax.set(aspect='equal')
        self.ax.axis("off")
        self.set_second(0)

    def set_sensors(self, sensor_pos):
        self.reset()
        circles = []
        for pos in sensor_pos.pos:
            c = plt.Circle(pos, radius=self.cr, facecolor="white",
                           linewidth=self.lw, edgecolor="black")
            self.ax.add_artist(c)
            circles.append(c)
        self.circles_ = circles
        xmin, xmax = np.min(sensor_pos.pos[:, 0]), np.max(sensor_pos.pos[:, 0])
        ymin, ymax = np.min(sensor_pos.pos[:, 1]), np.max(sensor_pos.pos[:, 1])
        if self.head_outline:
            s = 2 * self.cr
            c = plt.Circle((0, 0),
                           radius=max(abs(xmax-xmin), abs(ymax-ymin))/2+s,
                           facecolor="white",
                           linewidth=5, edgecolor="black", zorder=0)
            self.ax.add_artist(c)
            xmin, xmax = xmin-s, xmax+s
            ymin, ymax = ymin-s, ymax+s
        self.ax.set_xlim([xmin-abs(self.cr*2), xmax+abs(self.cr*2)])
        self.ax.set_ylim([ymin-abs(self.cr*2), ymax+abs(self.cr*2)])

    def set_second(self, seconds):
        s = "{:.1f}s".format(float(seconds))
        if self.title_ is None:
            self.title_ = self.ax.annotate(
                s,
                xy=(0.5, 0.95),
                xycoords="figure fraction",
                ha="center",
                va="center",
                size=self.fig.get_size_inches()[0]*3
            )
        else:
            self.title_.set_text(s)

    def set_colors(self, colors):
        for color, circle in zip(colors, self.circles_):
            circle.set_facecolor(color)

    def set_legend(self):
        pass


def _animation_func(frame, head_fig, predictions, msec_per_period):
    """ Animation helper func, called once per frame. Updates the figure """
    msec = frame * msec_per_period + msec_per_period
    head_fig.set_colors(predictions[frame])
    head_fig.set_second(msec / 1000)
    if not frame % 100:
        print("Frame {}".format(frame))


def animate_predictions(out_path,
                        head_fig,
                        predictions,
                        dpi=300,
                        video_msec_per_pred=1000,
                        msec_per_pred=1000):
    """

    Args:
        out_path:
        head_fig:
        predictions:
        dpi:
        video_msec_per_pred:
        msec_per_pred:

    Returns:

    """
    print("Making animation of {} steps, original time/step {} and "
          "video time/step {}. Video will be {} seconds long.".format(
        len(predictions), msec_per_pred, video_msec_per_pred,
        (len(predictions) * video_msec_per_pred)/1000
    ))

    from matplotlib.animation import FuncAnimation
    FuncAnimation(fig=head_fig.fig,
                  func=_animation_func,
                  fargs=(head_fig,
                         defaults.vectorized_stage_colors(predictions),
                         msec_per_pred),
                  init_func=lambda: None,
                  frames=len(predictions),
                  interval=video_msec_per_pred).save(
        out_path,
        dpi=dpi
    )
