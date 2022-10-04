import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.transforms as transforms
from collections import defaultdict
from glob import glob
from psg_utils import Defaults
from psg_utils.visualization.channels import (SensorPositions, HeadFigure,
                                               animate_predictions)

logger = logging.getLogger(__name__)


def load_subject(subject_paths):
    preds = np.stack([np.load(p) for p in subject_paths]).T
    return preds


def extract_channel(file_path):
    return file_path.split("pred_")[-1].split("/")[0]


def extract_channels(file_paths):
    return [extract_channel(p) for p in file_paths]


def load_all(glob_path):
    paths = glob(
        glob_path
    )
    # Load and separate by subject
    paths_by_subject = defaultdict(list)
    for path in paths:
        subject = os.path.split(path)[-1].replace("_PRED.npy", "")
        paths_by_subject[subject].append(path)
    # Order all equally
    channels = None
    for subject, file_paths in paths_by_subject.items():
        paths = sorted(file_paths, key=extract_channel)
        if channels is None:
            channels = extract_channels(paths)
    # Load all files
    return {
        subj: (load_subject(paths),) for subj, paths in paths_by_subject.items()
    }, channels


def group(subject_dict):
    arrs = list(zip(*subject_dict.values()))
    return list(map(np.vstack, arrs))


def reshape_to_epochs(preds, ms_per_pred, epoch_length_ms):
    folds = epoch_length_ms / ms_per_pred
    assert folds == int(folds)
    return preds.reshape(-1, int(folds), preds.shape[-1])


def add_labels(data_per_subject, glob_pattern):
    label_paths = np.array(glob(
        glob_pattern
    ))
    assert len(data_per_subject) == len(label_paths)
    for path in label_paths:
        subject = os.path.split(os.path.split(path)[0])[1]
        data_per_subject[subject] = (
            data_per_subject[subject][0],
            np.load(path)["arr_0"]
        )
    return data_per_subject


def group_by_true_labels(preds, labels):
    labels = labels.ravel()
    return [preds[labels == i] for i in range(5)]


def get_patterns(preds):
    c1, c2 = np.unique(preds, axis=0, return_counts=True)
    order = c2.argsort()[::-1]
    c1 = c1[order]
    c2 = c2[order]
    return c1, c2


def plot_pattern(pattern, channel_names=None, ms_per_step=None, title=None):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis('off')
    colored_pattern = defaults.vectorized_stage_colors(pattern)
    if title:
        fig.suptitle(title, fontsize=18)

    for chan_idx, channel in enumerate(colored_pattern.T):
        for col_idx, value in enumerate(channel):
            if col_idx == 0 and channel_names:
                trans = transforms.blended_transform_factory(ax.transAxes,
                                                             ax.transData)
                ax.annotate(
                    s=channel_names[chan_idx],
                    xy=(0, chan_idx),
                    xycoords=trans,
                    va="center",
                    ha="right"
                )
            if chan_idx == 0 and ms_per_step:
                trans = transforms.blended_transform_factory(ax.transData,
                                                             ax.transAxes)
                time = (ms_per_step/1000 * (col_idx+1) - ms_per_step / 2000)
                ax.annotate(
                    s="$T_c={:.1f}s$".format(time),
                    xy=(col_idx, 1.025),
                    xycoords=trans,
                    va="bottom",
                    ha="center"
                )
            plt.scatter([col_idx], [chan_idx], s=0)
            c = plt.Circle((col_idx, chan_idx),
                           radius=0.2,
                           facecolor=value,
                           linewidth=0.5,
                           edgecolor="black",
                           zorder=0)
            ax.add_artist(c)
    return fig, ax


def load_default_data():
    in_dir = '/Users/mathiasperslev/OneDrive/University/' \
             'phd/data/sleep/splits'
    glob_path = os.path.join(
        in_dir, "*", "predictions", "*", "high_res", "*.npy"
    )

    dat, channels = load_all(glob_path)
    channels = [c.replace("_", " + ") for c in channels]

    return dat, channels, in_dir


if __name__ == "__main__":
    dat, channels, in_dir = load_default_data()
    out_dir = in_dir + "/top_50_patterns"

    # Make animation?
    # sensors = SensorPositions(keep_channels=[c.split('-')[0] for c in channels])
    # head = HeadFigure(sensors, head_outline=False)
    # animate_predictions("/Users/mathiasperslev/Desktop/3s.mp4",
    #                     head_fig=head,
    #                     predictions=dat['tpcb68ff6d_a5c7_469d_ad8e_d90333a10337'],
    #                     video_msec_per_pred=100,
    #                     msec_per_pred=3000)

    # Settings
    ms_per_pred = 3000
    epoch_length_ms = 30*1000

    # Add labels for each subject?
    pattern = os.path.join(
        in_dir, "split_*", "predictions/pred_C3-M2_E2-M2/"
                           "test_data/files/*/true.npz"
    )
    dat = add_labels(dat, pattern)

    # TEMP
    preds = dat['tpcb68ff6d_a5c7_469d_ad8e_d90333a10337'][0]

    # preds, labels = group(dat)
    preds = reshape_to_epochs(preds,
                              ms_per_pred=ms_per_pred,
                              epoch_length_ms=epoch_length_ms)
    # preds = group_by_true_labels(preds, labels)

    for i in range(1500, 1600):
        plot_pattern(preds[i], channels, ms_per_pred)
        plt.show()

    stages = ["W", "N1", "N2", "N3", "REM"]
    for stage_idx, stage in enumerate(stages):
        print("Stage:", stage)
        # Make out directory
        out_dir_stage = os.path.join(out_dir, stage)
        os.mkdir(out_dir_stage)

        # Count interesting patterns...
        patterns, counts = get_patterns(preds[stage_idx])
        total = np.sum(counts)

        top_k = 50
        for i, (pattern, count) in enumerate(zip(patterns[:top_k],
                                                 counts[:top_k])):
            promil = count/total * 1000
            title = "Frequency = {:.1f}".format(promil) + \
                    u"\u2030" + " (rank {})".format(i+1)
            fig, ax = plot_pattern(pattern,
                                   channel_names=channels,
                                   ms_per_step=ms_per_pred,
                                   title=title)
            fig.savefig(out_dir_stage + "/pattern_{}.pdf".format(i+1))
            plt.close(fig)
