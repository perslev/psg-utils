"""
Implements the SleepStudy class which represents a sleep study (PSG)
"""

import logging
import numpy as np
from typing import Union
from psg_utils import errors, Defaults
from psg_utils.dataset.sleep_study.abc_sleep_study import AbstractBaseSleepStudy
from psg_utils.io.channels import ChannelMontageTuple, RandomChannelSelector
from psg_utils.io.high_level_file_loaders import get_org_include_exclude_channel_montages
from psg_utils.time_utils import TimeUnit

logger = logging.getLogger(__name__)


class H5SleepStudy(AbstractBaseSleepStudy):
    """
    Represents a PSG sleep study and (optionally) a manually scored hypnogram
    """
    def __init__(self,
                 h5_study_object,
                 annotation_dict=None,
                 no_hypnogram=False,
                 period_length: [int, float] = 30,
                 time_unit: Union[TimeUnit, str] = TimeUnit.SECOND,
                 internal_time_unit: Union[TimeUnit, str] = TimeUnit.MILLISECOND):
        """
        TODO
        """
        self.h5_study_object = h5_study_object
        super(H5SleepStudy, self).__init__(
            annotation_dict=annotation_dict,
            no_hypnogram=no_hypnogram,
            period_length=period_length,
            time_unit=time_unit,
            internal_time_unit=internal_time_unit,
            on_overlapping="RAISE"  # TODO - not used
        )
        if self.annotation_dict:
            self.annotation_dict = np.vectorize(annotation_dict.get)
        self._access_time_random_channel_selector = None
        self._n_classes = None  # Set in self.load
        self.load()  # Sets data visibility

    @property
    def identifier(self):
        """
        Returns an ID, which is simply the name of the directory storing
        the data
        """
        return self.h5_study_object.name.split("/")[-1]

    def __str__(self):
        if self.loaded:
            t = (self.loaded, self.open, self.identifier,
                 len(self.select_channels),self.sample_rate,
                 self.hypnogram is not False)
            return "H5SleepStudy(open={}, loaded={}, identifier={:s}, " \
                   "N channels: {}, sample_rate={:.1f}, " \
                   "hypnogram={})".format(*t)
        else:
            return repr(self)

    def __repr__(self):
        return 'H5SleepStudy(open={}, loaded={}, identifier={})' \
               ''.format(self.open, self.loaded, self.identifier)

    @property
    def loaded(self):
        """
        Returns whether the PSG and hypnogram properties are set or not.
        Only affects 'visibility' to the data, no data is actually loaded.
        """
        return not any((self.psg is None,
                        self.hypnogram is None))

    @property
    def open(self):
        """
        Returns whether the HDF5 file is currently open or not
        """
        return bool(self.h5_study_object)

    def _load_with_any_in(self, channel_sets, channels_in_file, allow_missing_channels: bool = False):
        """
        TODO

        Args:
            channel_sets:
            channels_in_file:

        Returns:

        """
        for i, channel_set in enumerate(channel_sets):
            try:
                # Work out which channels to include and exclude during loading
                org_channels, include_channels, _, _ = \
                    get_org_include_exclude_channel_montages(
                        load_channels=channel_set,
                        header={'channel_names': channels_in_file},
                        allow_missing_channels=allow_missing_channels
                    )
                return include_channels
            except errors.ChannelNotFoundError as e:
                if i < len(channel_sets) - 1:
                    # Try nex set of channels
                    continue
                else:
                    s, sa = self.select_channels, \
                            self.alternative_select_channels
                    err = errors.ChannelNotFoundError("Could not load "
                                                      "select_channels {} or "
                                                      "alternative_select_"
                                                      "channels "
                                                      "{}".format(s, sa))
                    raise err from e

    def load(self, allow_missing_channels=False):
        """
        Sets the PSG and hypnogram visibility according to self._try_channels.
        """
        # Get channels
        channels = ChannelMontageTuple(list(self.h5_study_object['PSG'].keys()))
        loaded_channels = self._load_with_any_in(self._try_channels,
                                                 channels_in_file=channels,
                                                 allow_missing_channels=allow_missing_channels)
        self._psg = {
            chan: self.h5_study_object['PSG'][chan.original_name] for chan in loaded_channels
        }
        self._set_loaded_channels(loaded_channels)
        self._hypnogram = np.asarray(self.h5_study_object['hypnogram'])
        self._n_classes = len(np.unique(self._hypnogram))
        self._class_to_period_dict = {str(class_int): np.asarray(class_indices) for class_int, class_indices
                                      in self.h5_study_object['class_to_index'].items()}

    def unload(self):
        """ Sets the PSG and hypnogram properties to None """
        self._psg = None
        self._hypnogram = None
        self._class_to_period_dict = None

    def reload(self, warning=True, allow_missing_channels=False):
        """ Only sets the current channel visibility """
        self.unload()
        self.load(allow_missing_channels=allow_missing_channels)

    def get_psg_shape(self) -> tuple:
        """
        Returns the shape of the PSG array as returned by self.get_psg_as_array

        Returns:
            tuple, [n_periods * data_per_period, self.n_channels]
        """
        return len(self.psg[self.select_channels[0]]), self.n_channels

    def get_psg_as_array(self) -> np.ndarray:
        """
        Returns the PSG stored in self.psg as a single, flat ndarray of shape [-1, n_channels]
        """
        psg = np.empty(shape=self.get_psg_shape(), dtype=Defaults.PSG_DTYPE)
        for i, c in enumerate(self.select_channels):
            psg[:, i] = self.psg[c]
        return psg

    def get_class_indices(self, class_int: int) -> np.ndarray:
        """
        TODO

        Args:
            class_int:

        Returns:

        """
        return self._class_to_period_dict[str(class_int)]

    def translate_labels(self, y):
        """
        TODO

        Returns:

        """
        if self.annotation_dict:
            return self.annotation_dict(y)
        else:
            return y

    def _get_sample_channels(self) -> list:
        """
        If a RandomChannelSelector is not set (see self.access_time_random_channel_selector setter method), simply
        returns the self.select_channels property. Otherwise, samples a set of channels from the RandomChannelSelector
        (passing in channels in self.select_channels to select from) and returns those.

        Returns:
            channels: a list of channels
        """
        if not self.access_time_random_channel_selector:
            return self.select_channels
        else:
            return self.access_time_random_channel_selector.sample(
                available_channels=self.select_channels
            )

    def get_psg_periods_by_idx(self, start_idx: int, n_periods: int = 1, channel_indices: list = None) -> np.ndarray:
        """
        Returns periods from the PSG in shape [n_periods, self.data_per_period, n_channels].

        Args:
            start_idx (int):        Index of first period to return
            n_periods (int):        The number of periods to return
            channel_indices (list): Optional list of channel indices to extract from PSG. Extracts all with None.

        Returns:
            psg: ndarray of shape [n_periods, self.data_per_period, n_channels]
        """
        if channel_indices is not None:
            # TODO consider changing ABC; Use 'channel_names' instead of 'channel_indices' across study classes.
            # TODO consider interaction between RandomChannelSelector and future 'channel_names' argument
            self.raise_err(NotImplementedError, f"Cannot specify parameter 'channel_indices' ({channel_indices}) "
                                                f"with this study class.")
        # Sample (if RandomChannelSelector is set) or get select_channels property
        channels = self._get_sample_channels()
        psg = np.empty(shape=[n_periods * self.data_per_period, len(channels)], dtype=Defaults.PSG_DTYPE)
        data_start_idx = start_idx * self.data_per_period
        data_end_idx = data_start_idx + (self.data_per_period * n_periods)
        for i, chan in enumerate(channels):
            psg[:, i] = self.psg[chan][data_start_idx:data_end_idx]
        return psg.reshape([n_periods, self.data_per_period, psg.shape[-1]])

    def get_hyp_periods_by_idx(self, start_idx: int, n_periods: int = 1, on_overlapping: [str, None] = None) -> np.ndarray:
        """
        Returns periods from the hypnogram in shape [n_periods].

        Args:
            start_idx (int): Index of first period to return
            n_periods (int): The number of periods to return
            on_overlapping:  Not used

        Returns:
            hyp: ndarray of shape [n_periods]
        """
        hyp = self.hypnogram[start_idx:start_idx+n_periods]
        return self.translate_labels(hyp).reshape([-1])

    @property
    def sample_rate(self) -> int:
        """
        Returns the sample rate as an integer
        """
        return int(self.h5_study_object.attrs.get('sample_rate'))

    @property
    def date(self):
        """
        TODO

        Returns:

        """
        return self.h5_study_object.attrs.get('date')

    @property
    def n_classes(self) -> int:
        """
        Returns the number of unique classes in self.hypnogram
        """
        return self._n_classes

    @property
    def n_sample_channels(self):
        if self.access_time_random_channel_selector:
            return self.access_time_random_channel_selector.n_output_channels
        else:
            return self.n_channels

    @property
    def access_time_random_channel_selector(self):
        """
        TODO

        Returns:

        """
        return self._access_time_random_channel_selector

    @access_time_random_channel_selector.setter
    def access_time_random_channel_selector(self, channel_selector):
        """
        TODO

        Args:
            channel_selector:

        Returns:

        """
        if channel_selector is not None and not \
                isinstance(channel_selector, RandomChannelSelector):
            raise TypeError("Expected 'channel_selector' argument to be of "
                            "type {}, got {}".format(type(RandomChannelSelector),
                                                     type(channel_selector)))
        self._access_time_random_channel_selector = channel_selector

    def to_batch_generator(self, batch_size, overlapping=False):
        """
        Yields batches of data from the SleepStudy PSG/HYP pair
        Note: With overlapping == False the last batch may be smaller than
        batch_size due to boundary effects.

        Args:
            batch_size:  An integer, number of periods/epochs/segments to
                         return in each batch.
            overlapping: Yield overlapping batches (sliding window). Otherwise
                         return non-overlapping, connected segments.

        Yields:
            X: ndarray of shape [batch_size, self.data_per_period,
                                 self.n_channels]
            y: ndarray of shape [batch_size, 1]
        """
        # TODO
        raise NotImplementedError("TODO")
        if overlapping:
            raise NotImplementedError("H5SleepStudy objects do not support "
                                      "to_batch_generator with "
                                      "overlapping=True yet.")
        end_point = self.n_periods-(self.n_periods % batch_size)
        for idx in range(0, end_point, batch_size):
            yield self.get_periods_by_idx(
                start_idx=idx,
                end_idx=idx+batch_size-1
            )
