import logging
import numpy as np
import h5py
import time
from datetime import datetime
from psg_utils.io.channels.channels import ChannelMontageTuple

logger = logging.getLogger(__name__)


def to_ids(start, durs, stage, out):
    """
    Save init/start, durs, stage lists to .ids format

    :param start: list of ints/floats, list of start/onset/inits
    :param durs: list of ints/float, list of durations
    :param stage: list, list of stages (typically string or ints)
    :param out: str, path to output path, usually with suffix '.ids'
    """
    with open(out, "w") as out_f:
        for i, d, s in zip(start, durs, stage):
            out_f.write("{},{},{}\n".format(i, d, s))


def to_h5_file(out_path, data, channel_names, sample_rate, date, dtype=np.float32, **kwargs):
    """
    Saves a NxC ndarray 'data' of PSG data (N samples, C channels) to a .h5
    archive at path 'out_path'. A list 'channel_names' of length C must be
    passed, giving the name of each channel in 'data'. Each Nx1 array in 'data'
    will be stored under groups in the h5 archive according to the channel name

    Also sets h5 attributes 'date' and 'sample_rate'.

    Args:
        out_path:      (string)   Path to a h5 archive to write to
        data:          (ndarray)  A NxC shaped ndarray of PSG data
        channel_names: (list)     A list of C strings giving channel names for
                                  all channels in 'data'
        sample_rate:   (int)      The sample rate of the signal in 'data'.
        date:          (datetime) A datetime object. Is stored as a timetuple
                                  within the archive. If a non datetime object
                                  is passed, this will be stored 'as-is'.
        dtype:         (np.dtype) The datatype to store the data as
    """
    if len(data.shape) != 2:
        raise ValueError("Data must have exactly 2 dimensions, "
                         "got shape {}".format(data.shape))
    if data.shape[-1] == len(channel_names):
        assert data.shape[0] != len(channel_names)  # Should not happen
        data = data.T
    elif data.shape[0] != len(channel_names):
        raise ValueError("Found inconsistent data shape of {} with {} select "
                         "channels ({})".format(data.shape,
                                                len(channel_names),
                                                channel_names))
    if isinstance(date, datetime):
        # Convert datetime object to TS naive unix time stamp
        date = time.mktime(date.timetuple())
    if isinstance(channel_names, ChannelMontageTuple):
        channel_names = channel_names.original_names
    data = data.astype(dtype)
    with h5py.File(out_path, "w") as out_f:
        out_f.create_group("channels")
        for i, (chan_dat, chan_name) in enumerate(zip(data, channel_names)):
            dataset = out_f['channels'].create_dataset(
                chan_name,
                data=chan_dat,
                chunks=True,
                compression='gzip'
            )
            dataset.attrs["channel_index"] = i
        out_f.attrs['date'] = date or "UNKNOWN"
        out_f.attrs["sample_rate"] = sample_rate
