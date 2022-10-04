"""
A set of functions that apply quality-control checks on the PSG data.
The function take in a PSG and a set of parameters and may alter the PSG.

The period_length argument should also be accepted, but functions could
ignore this and do quality control checks that exceed the original epoch
boundaries.
"""
import logging
import numpy as np
from typing import Tuple, List
from psg_utils.errors import NotLoadedError
from psg_utils.time_utils import TimeUnit

logger = logging.getLogger(__name__)


def assert_no_nans(psg):
    if np.any(np.isnan(psg)):
        raise ValueError("Cannot compute IQR values for PSG array which contains one or more NaN values.")


def zero_out_noisy_epochs(psg, sample_rate, period_length_sec,
                          max_times_global_iqr=20) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Sets all values in a epoch of 'period_length' seconds of signal to zero
    (channel-wise) if any (absolute) value within that period exceeds
    'max_times_global_iqr' times the IQR of all data in the channel across time

    Operates in-place.

    TODO: Refactor max_times_global_iqr to match clip_noisy_values

    Args:
        psg:                  A ndarray of shape [N, C] of PSG data
        sample_rate:          The sample rate of data in the PSG
        period_length_sec:    The length of one epoch/period/segment in seconds
        max_times_global_iqr: Extreme value threshold; number of times an absolute value
                              in a channel must exceed the global IQR for that
                              channel for it to be termed an outlier.

    Returns:
        PSG, ndarray of shape [N, C]
        A list of lists, one sub-list for each channel, each storing indices
        of all epochs that were set to zero.
    """
    assert_no_nans(psg)
    n_channels = psg.shape[-1]
    chan_inds = []
    for chan in range(n_channels):
        chan_psg = psg[..., chan]

        # Compute global IQR
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * max_times_global_iqr

        # Reshape PSG to periods on 0th axis
        n_periods = int(chan_psg.shape[0]/(sample_rate*period_length_sec))
        chan_psg = chan_psg.reshape(n_periods, -1)

        # Compute IQR for all epochs
        inds = np.unique(np.where(np.abs(chan_psg) > threshold)[0])

        # Zero out noisy epochs in the particular channel
        chan_psg[inds] = 0.
        psg[:, chan] = np.reshape(chan_psg, [-1])
        chan_inds.append(inds)
    return psg, chan_inds


def clip_noisy_values(psg, sample_rate, period_length_sec,
                      min_max_times_global_iqr=20) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Clips all values that are larger or smaller than +- min_max_times_global_iqr
    times to IQR of the whole channel.

    Operates in-place.

    TODO: Refactor min_max_times_global_iqr to match zero_out_noisy_epochs

    Args:
        psg:                      A ndarray of shape [N, C] of PSG data
        sample_rate:              The sample rate of data in the PSG
        period_length_sec:        The length of one epoch/period/segment in
                                  seconds
        min_max_times_global_iqr: Extreme value threshold; number of times an absolute
                                  value in a channel must exceed the global IQR
                                  for that channel for it to be termed an
                                  outlier.

    Returns:
        PSG, ndarray of shape [N, C]
        A list of lists, one sub-list for each channel, each storing indices
        of all epochs in which one or more values were clipped.
    """
    assert_no_nans(psg)
    n_channels = psg.shape[-1]
    chan_inds = []
    for chan in range(n_channels):
        chan_psg = psg[..., chan]

        # Compute global IQR
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * min_max_times_global_iqr

        # Reshape PSG to periods on 0th axis
        n_periods = int(chan_psg.shape[0]/(sample_rate*period_length_sec))
        temp_psg = chan_psg.reshape(n_periods, -1)

        # Compute IQR for all epochs
        inds = np.unique(np.where(np.abs(temp_psg) > threshold)[0])
        chan_inds.append(inds)

        # Zero out noisy epochs in the particular channel
        psg[:, chan] = np.clip(chan_psg, -threshold, threshold)
    return psg, chan_inds


def apply_quality_control_func(sleep_study, sample_rate, warn_fraction=0.15, warn=True):
    """
    Applies the quality control function set on a SleepStudy object to itself.

    Operates in-place on the sleep_study.psg object.

    Args:
        sleep_study:    A SleepStudy object
        sample_rate:    The sample rate of the currently set PSG.
        warn_fraction:  If the fraction of epochs affected by the quality control function
                        is >= 'warn_fraction' a logger warning is issued.
                        Otherwise, a debug logging is issued.
        warn            Whether to warn on warn_fraction exceeded. If False, do not warn no matter
                        the fraction of epochs affected by QA.

    Returns:
        The PSG ndarray object to which QA has been applied.
    """
    if not sleep_study.loaded:
        raise NotLoadedError("Cannot apply quality control func to {} "
                             "as it is not loaded.".format(sleep_study))
    if not sleep_study.quality_control_func:
        raise TypeError("Cannot apply quality control function to {} as its "
                        "quality_control_func argument is not "
                        "set.".format(sleep_study))
    func_str, kwargs = sleep_study.quality_control_func
    f = globals()[func_str]
    period_length_sec = sleep_study.get_period_length_in(TimeUnit.SECOND)
    psg, inds = f(psg=sleep_study.psg,
                  sample_rate=sample_rate,
                  period_length_sec=period_length_sec,
                  **kwargs)
    if warn:
        n_periods = int(psg.shape[0] / (sample_rate * period_length_sec))
        for i, chan_inds in enumerate(inds):
            fraction = len(chan_inds) / n_periods
            warn_str = "Quality control for sample '{}' affected " \
                       "{}/{} epochs in channel {}".format(sleep_study.identifier or "<identifier not passed>",
                                                           len(chan_inds), n_periods, i)
            logger.warning(warn_str) if fraction >= warn_fraction else logger.debug(warn_str)
    return psg
