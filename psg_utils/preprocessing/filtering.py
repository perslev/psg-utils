import numpy as np
from mne.filter import filter_data, notch_filter


def apply_filtering(psg, sample_rate, **filter_kwargs) -> np.ndarray:
    """
    Applies the mne.filter.filter_data method on PSG array (ndarray, [N, C]) with
    parameters as specified by filter_kwargs.

    Example parameters for 0.3-35 Hz band-pass:
    filter_kwargs: {'l_freq': 0.3, 'h_freq': 35, 'method': 'fir'}

    Args:
        psg:                      A ndarray of shape [N, C] of PSG data
        sample_rate:              The sample rate of data in the PSG
        **filter_kwargs:          Filtering arguments passed to mne.filter.filter_data
    """
    dtype_mem = psg.dtype
    return filter_data(
        psg.T.astype(np.float64), sample_rate, **filter_kwargs
    ).T.astype(dtype_mem)


def apply_notch_filtering(psg, sample_rate, **notch_filter_kwargs) -> np.ndarray:
    """
    Applies the mne.filter.notch_filter method on PSG array (ndarray, [N, C]) with
    parameters as specified by filter_kwargs.

    Example parameters for 50 Hz notch filter:
    filter_kwargs: {'freqs': 50}

    Args:
        psg:                      A ndarray of shape [N, C] of PSG data
        sample_rate:              The sample rate of data in the PSG
        **notch_filter_kwargs:    Filtering arguments passed to mne.filter.notch_filter
    """
    dtype_mem = psg.dtype
    return notch_filter(
        psg.T.astype(np.float64), sample_rate, **notch_filter_kwargs
    ).T.astype(dtype_mem)
