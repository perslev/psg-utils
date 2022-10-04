import numpy as np
import matplotlib.pyplot as plt
from utime import Defaults
from matplotlib import gridspec, transforms
from psg_utils.io.channels import ChannelMontageTuple


def get_figure(n_channels, n_pred, n_annotations=1):
    # hr = [(1-(pred_fraction if ))/n_channels]*n_channels + ([pred_fraction] * n_pred)
    annot_height = 0.55 * n_annotations + (0.30 if n_annotations else 0)
    fig_height = 5.5 + annot_height + 1.5 * (n_channels + n_pred)
    top = 1 - (0.5+annot_height)/fig_height
    bot = 2.0/fig_height
    gs = gridspec.GridSpec(n_channels + n_pred, 1, hspace=0, bottom=bot, top=top, left=0.05, right=0.85)
    figure = plt.figure(num=1, figsize=(25, fig_height))
    axes = [figure.add_subplot(i) for i in gs]
    data_axes = axes[:n_channels]
    for i in range(len(data_axes)-1):
        data_axes[i].spines['bottom'].set_visible(False)
        data_axes[i+1].spines['top'].set_visible(False)
    pred_axes = axes[n_channels:]
    return figure, data_axes, pred_axes


def set_seconds_ticks(ax, sample_rate, period_length_sec, n_periods):
    ticks = [sample_rate * period_length_sec * i for i in range(n_periods + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick//sample_rate)) + "s" for tick in ticks], rotation=90)


def plot_pred_data(pred, figure, pred_ax, sample_rate, n_periods, period_length_sec=30, add_labels=True, add_legend=True):
    assert pred.ndim == 2
    pred = pred/np.sum(pred, axis=1, keepdims=True)

    axis_length = sample_rate * period_length_sec * n_periods
    xs = np.arange(axis_length)
    if pred.shape[0] != axis_length:
        pred = np.repeat(pred, int(axis_length/pred.shape[0]), axis=0)

    # Plot
    stage_map = Defaults.get_class_int_to_stage_string()
    pred_ax.stackplot(xs, pred.T,
                      labels=[stage_map[i] for i in range(pred.shape[-1])],
                      colors=Defaults.STAGE_COLORS, alpha=0.65)
    pred_ax.set_xlabel("Time (s)", size=22, labelpad=10)
    pred_ax.set_xlim(xs[0], xs[-1])
    pred_ax.set_ylim(0, 1)
    pred_ax.set_yticks([0.5]), pred_ax.set_yticklabels([0.5])
    if add_labels:
        set_seconds_ticks(pred_ax, sample_rate, period_length_sec, n_periods)
    elif not add_labels:
        pred_ax.set_xticks([]), pred_ax.set_xticklabels([])

    # Add legend
    if add_legend:
        l = pred_ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.2/figure.get_size_inches()[1]),
                           ncol=pred.shape[-1],
                           fontsize=21,
                           bbox_transform=figure.transFigure)
        l.get_frame().set_linewidth(0)


def plot_psg_data(psg, sample_rate, preds=None, annotations={}, data_per_period=None, channel_names=None, period_length_sec=30):
    n_channels = psg.shape[-1]
    n_periods = psg.shape[0]
    if preds is not None and isinstance(preds, np.ndarray) and preds.ndim == 2:
        preds = [preds]
    if preds is not None:
        if data_per_period is None:
            raise ValueError("Must specify data_per_period when 'preds' is passed.")
        if not isinstance(data_per_period, (tuple, list, np.ndarray)):
            data_per_period = [data_per_period]
        assert len(preds) == len(data_per_period)
    n_pred = 0 if preds is None else len(preds)
    figure, data_axes, pred_axes = get_figure(n_channels=n_channels, n_pred=n_pred, n_annotations=len(annotations))
    psg_flat = psg.reshape(-1, psg.shape[-1])

    # Compute axis limits
    xs = np.arange(len(psg_flat))
    xlim = np.min(xs), np.max(xs)
    ylims = [
        (np.min(psg[..., i]),
         np.max(psg[..., i]))
        for i in range(n_channels)
    ]
    ylims = [(-np.max(np.abs(y)), np.max(np.abs(y))) for y in ylims]

    # Plot PSG data channel-by-channel and set xlims, ticks and ticklabels
    for i, (psg_chan, ax, ylim) in enumerate(zip(psg_flat.T, data_axes, ylims)):
        ax.plot(xs, psg_chan, color="black", linewidth=1.5)
        ax.axhline(0, linewidth=1, linestyle="-",
                   color="grey", zorder=-2)
        ax.set_xticks([]), ax.set_xticklabels([])
        ax.set_yticks([0]), ax.set_yticklabels(["0 µV"])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Annotate
    d = len(psg[0])
    ticks = [d * i for i in range(n_periods)]
    stage_positions = [d // 2 + (d * i) for i in range(n_periods)]
    for t in ticks:
        for ax in data_axes:
            ax.axvline(t, linewidth=1, linestyle='-',
                       color="grey",
                       alpha=0.95, zorder=-1)

    # Set sleep stages by different annotators above the signal plot
    for i, (label, hyp) in enumerate(annotations.items()):
        hyp_flat = np.array(hyp).flatten()
        label_map = Defaults.get_class_int_to_stage_string()
        labels = [label_map[int(y)] for y in hyp_flat]
        colors = [Defaults.STAGE_COLORS[y] for y in hyp_flat]
        for sp, lab, col in zip(stage_positions, labels, colors):
            label_y = 1.20 + 0.25 * i
            data_axes[0].annotate(text=lab, xy=(sp, label_y),
                                  xycoords=transforms.blended_transform_factory(data_axes[0].transData, data_axes[0].transAxes),
                                  size=max(4.0, 40 - n_periods*0.8),
                                  ha="center", va="center", color=col)
        data_axes[0].annotate(
            text=label,
            size=20,
            xy=(1.02, label_y),
            xycoords=transforms.blended_transform_factory(data_axes[0].transAxes, data_axes[0].transAxes),
            va="center",
            ha="left"
        )

    # Add channel names to right-side of data plots
    if channel_names is not None:
        assert len(channel_names) == n_channels
        if isinstance(channel_names, ChannelMontageTuple):
            channel_names = channel_names.original_names
        for (ax, channel) in zip(data_axes, channel_names):
            ax.annotate(
                text=channel,
                size=20,
                xy=(1.02, 0.5),
                xycoords=ax.transAxes,
                va="center",
                ha="left"
            )

    if preds is not None:
        for i, pred in enumerate(preds):
            annotate = i == len(preds) - 1
            plot_pred_data(pred, figure, pred_axes[i], sample_rate, n_periods, period_length_sec=period_length_sec, add_labels=annotate, add_legend=annotate)
            # Add model confidence label
            fq = str(sample_rate / data_per_period[i])
            ax.annotate(text=u"\u2248" + f"{fq[:fq.index('.') + 3]} Hz",
                        size=20,
                        xy=(1.02, 0.5),
                        xycoords=pred_axes[i].transAxes,
                        va="center",
                        ha="left"
                        )
    else:
        set_seconds_ticks(data_axes[-1], sample_rate, period_length_sec, n_periods)

    # Ax label size
    for ax in data_axes + pred_axes:
        ax.tick_params(axis='both', which='major', labelsize=14)

    return figure, data_axes, pred_axes
