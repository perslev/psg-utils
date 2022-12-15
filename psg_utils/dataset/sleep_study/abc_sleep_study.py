import logging
import numpy as np
import math
from typing import Tuple, Union, List
from abc import ABC, abstractmethod
from contextlib import contextmanager
from psg_utils import Defaults
from psg_utils.time_utils import TimeUnit, convert_time, standardize_time_input
from psg_utils.utils import ensure_list_or_tuple
from datetime import datetime


logger = logging.getLogger(__name__)


class AbstractBaseSleepStudy(ABC):
    """
    TODO
    """
    def __init__(self,
                 annotation_dict: dict,
                 no_hypnogram: bool,
                 period_length: [int, float],
                 time_unit: Union[TimeUnit, str],
                 internal_time_unit: Union[TimeUnit, str],
                 on_overlapping: str):
        """
        Args:
           no_hypnogram       (bool)       Initialize without ground truth data.
           annotation_dict    (dict)       A dictionary mapping from labels in the hyp_file_path file to integers
           period_length      (int/float)  Sleep 'epoch' (segment/period) length in units 'time_unit' (see below)
           time_unit          (TimeUnit)   TimeUnit object specifying the unit of time of 'period_length'
           internal_time_unit (TimeUnit)   TimeUnit object specifying the unit of time to use internally for storing
                                             times. Affects the values returned by methods or attributes such as
                                             self.period_length.
           on_overlapping:    (str)        One of 'FIRST', 'LAST', 'MAJORITY', 'RAISE'.
                                             Controls the behaviour when a discrete period of length self.period_length
                                             overlaps 2 or more different classes in the original hypnogram.
                                             See SparseHypnogram.get_period_at_time for details.
        """
        self.annotation_dict = annotation_dict
        self._no_hypnogram = bool(no_hypnogram)

        # Set the on_overlapping property for when/if hypnogram in loaded.
        # Has no effect if no_hypnogram is True.
        on_overlapping = on_overlapping.upper()
        if on_overlapping not in ('FIRST', 'LAST', 'MAJORITY', 'RAISE'):
            raise ValueError(f"Got unexpected value for parameter 'on_overlapping' ({on_overlapping}). "
                             f"Expected one of 'FIRST', 'LAST', 'MAJORITY', 'RAISE'.")
        self.on_overlapping = on_overlapping

        # Convert period_length input to an internal integer representation in units 'internal_time_unit' (TimeUnit)
        self._time_unit = standardize_time_input(internal_time_unit)
        self._data_time_unit = standardize_time_input(time_unit)
        try:
            self._period_length = convert_time(period_length, self.data_time_unit, self.time_unit, cast_to_int=True)
        except ValueError as e:
            raise ValueError(f"Parameter 'period_length' should be a whole number/integer. "
                             f"Consider setting different org and/or internal time units "
                             f"(e.g., if you want to use a period_length of 2.5 milliseconds, "
                             f"set period_length=2.5, org_time_unit=TimeUnit.MILLISECONDS and "
                             "internal_time_unit=TimeUnit.MICROSECONDS.") from e

        # Hidden attributes controlled in property functions to limit setting
        # of these values to the load() function
        self._psg = None
        self._hypnogram = None
        self._class_to_period_dict = None
        self._select_channels = None
        self._alternative_select_channels = None

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def identifier(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def loaded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reload(self, warning, allow_missing_channels) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, allow_missing_channels) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_class_indices(self, class_int: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_psg_shape(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def get_psg_as_array(self):
        """
        Returns the PSG stored in self._psg as a single, flat ndarray of shape [-1, n_channels]
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def get_hyp_periods_by_idx(self, start_idx: int, n_periods: int = 1, on_overlapping: Union[str, None] = None) -> np.ndarray:
        """
        Returns periods from the hypnogram in shape [n_periods].

        Args:
            start_idx (int): Index of first period to return
            n_periods (int): The number of periods to return
            on_overlapping: str or None, if str one of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a
                            discrete period of length self.period_length overlaps 2 or more different classes in the
                            original hypnogram. See SparseHypnogram.get_period_at_time for details. Default with
                            on_overlapping = None is self.on_overlapping.

        Returns:
            hyp: ndarray of shape [n_periods]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """ Returns the sample rate in Hertz. Must be a whole number. """
        raise NotImplementedError

    @property
    @abstractmethod
    def date(self) -> Union[None, datetime, str]:
        """ Returns the date. May be None, a datetime object or string (e.g. UNKNOWN) """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_classes(self) -> int:
        raise NotImplementedError

    @property
    def class_to_period_dict(self) -> dict:
        return self._class_to_period_dict

    @property
    def classes(self) -> np.ndarray:
        return np.array(sorted(self._class_to_period_dict.keys()))

    @property
    def no_hypnogram(self) -> bool:
        return self._no_hypnogram

    @property
    def time_unit(self) -> TimeUnit:
        """ Returns the internal time unit """
        return self._time_unit

    @property
    def data_time_unit(self) -> TimeUnit:
        """ Returns the time unit for the raw (hypnogram) data """
        return self._data_time_unit

    @property
    def period_length(self) -> int:
        """ Returns the period length in time unit self.time_units as an integer """
        return self._period_length

    def get_period_length_in(self, time_unit: Union[TimeUnit, str]) -> float:
        """ Returns self.period_length in time unit 'time_unit' """
        return convert_time(self.period_length, self.time_unit, time_unit)

    @property
    def recording_length(self) -> int:
        """ Returns the recording length in time unit self.time_units """
        return convert_time(self.get_psg_shape()[0] / self.sample_rate,
                            from_unit=TimeUnit.SECOND,
                            to_unit=self.time_unit,
                            cast_to_int=True)

    def get_recording_length_in(self, time_unit: Union[TimeUnit, str]) -> float:
        """ Returns the recording length in units 'time_unit' """
        return convert_time(self.recording_length, self.time_unit, time_unit)

    @property
    def last_period_start(self) -> int:
        """ Returns the recordings last period start time in unit self.time_unit """
        reminder = self.recording_length % self.period_length
        if reminder > 0:
            return self.recording_length - reminder
        else:
            return self.recording_length - self.period_length

    def get_last_period_start_in(self, time_unit: Union[TimeUnit, str]) -> float:
        """
        Returns the time that marks the beginning of the last period in units 'time_unit'
        """
        return convert_time(self.last_period_start, self.time_unit, time_unit)

    @property
    def n_periods(self) -> int:
        """
        Returns the number of periods of length self.period_length in the PSG.
        Note that this include any partial final period of less than self.period_length length.
        """
        return int(np.ceil(self.recording_length / self.period_length))

    @property
    def n_channels(self) -> int:
        """ Returns the number of channels in the PSG array """
        return len(self.select_channels)

    @property
    def n_sample_channels(self) -> int:
        """
        Overwritten in some derived classes that sample channels on-access
        Always returns int >=1, even if self.n_channels returns 0
        """
        return self.n_channels or 1

    def period_idx_to_time(self, period_idx: int) -> int:
        """
        Helper method that maps a period_idx (int) to the first time point in that period.
        """
        self._assert_period_index_bounds(period_idx)
        return period_idx * self.period_length

    def period_time_to_idx(self, time: [int, float], time_unit: Union[TimeUnit, str]) -> int:
        """
        Helper method that maps a period start time in units 'time_unit' to its period index.
        """
        time = self._to_internal_period_time(time, time_unit)
        return int(time / self.period_length)

    def _assert_similar_lengths(self, psg_arr, hyp_arr):
        """
        Helper method to check if the number of periods in a PSG array matches that of a HYP array
        Raises ValueError if len(psg_arr) != len(hyp_arr)
        """
        if len(psg_arr) != len(hyp_arr):
            err_msg = ("Length of PSG array does not match length dense "
                       "hypnogram array ({} != {}). If hypnogram "
                       "is longer, consider if a trailing or leading "
                       "sleep stage should be removed. (you may use "
                       "SleepStudyDataset.set_strip_func())".format(len(psg_arr),
                                                                    len(hyp_arr)))
            self.raise_err(ValueError, err_msg)

    def _assert_period_index_bounds(self, period_idx: int):
        if period_idx < 0 or self.n_periods <= period_idx:
            self.raise_err(IndexError, f"Period index {period_idx} is out of bounds "
                                       f"(either negative or >= the total number of periods "
                                       f"({self.n_periods})")

    def get_periods_by_idx(self,
                           start_idx: int,
                           n_periods: int = 1,
                           channel_indices: list = None,
                           on_overlapping: Union[str, None] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get a range of period of {X, y} data by indices (self.no_hypnogram is False) else {X}
        Period starting at second 0 is index 0.

        Args:
            start_idx       (int):  Index of first period to return
            n_periods       (int):  The number of periods to return
            channel_indices (list): Optional list of channel indices to extract from PSG. Extracts all with None.
            on_overlapping: str or None, if str one of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a
                            discrete period of length self.period_length overlaps 2 or more different classes in the
                            original hypnogram. See SparseHypnogram.get_period_at_time for details. Default with
                            on_overlapping = None is self.on_overlapping.

        Returns:
            psg: ndarray of shape [n_periods, self.data_per_period, C]
            hyp: ndarray of shape [n_periods, 1] IF self.no_hypnogram is False
        """
        psg = self.get_psg_periods_by_idx(start_idx, n_periods, channel_indices)
        if self.no_hypnogram:
            return psg
        else:
            hyp = self.get_hyp_periods_by_idx(start_idx, n_periods, on_overlapping)
            self._assert_similar_lengths(psg, hyp)
            return psg, hyp

    def _to_internal_period_time(self, time: [int, float], time_unit: Union[TimeUnit, str]) -> int:
        """
        Helper method that:

        (1) Converts a time point 'time' in units 'time_unit' to the internal time units self.time_unit
        (2) Checks if the time point aligns with the beginning of a period.
        (3) That 'time' is within bounds of the study (0 <= time < self.recording_length)

        Raises ValueError if the time point cannot be converted to internal integer representation,
        or (2) or (3) are violated.

        Args:
            time (int, float):           The time point of the beginning of a period.
            time_unit: (TimeUnit, str):  Time unit for parameter 'time'.

        Returns:
            internal_time: int, internal time aligning with beginning of a period.
        """
        try:
            internal_time = convert_time(time, time_unit, self.time_unit, cast_to_int=True)
        except ValueError as e:
            self.raise_err(ValueError, f"Cannot get PSG/hyp period starting at time {time} ({time_unit}) as the "
                                       f"time cannot be safely cast to an integer/whole number using the internal time "
                                       f"unit {self.time_unit}", _from=e)
        else:
            if internal_time % self.period_length:
                self.raise_err(ValueError, f"Invalid time of {internal_time}, not divisible by period "
                                           f"length of {self.period_length}")
            if internal_time >= self.recording_length or internal_time < 0:
                self.raise_err(IndexError, f"Invalid time of {internal_time}, outside range of sleep study "
                                           f"of {self.recording_length} ({self.time_unit})")
            return internal_time

    def get_psg_periods_by_time(self,
                                start_time: [int, float],
                                time_unit: Union[TimeUnit, str],
                                n_periods: int = 1,
                                channel_indices: list = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Equivalent method to self.get_psg_periods_by_idx but working in time units instead of period indices.

        Returns periods from the PSG in shape [n_periods, self.data_per_period, n_channels] starting
        from a time point 'start_time'. 'start_time' must align exactly with the beginning of a period, otherwise a
        ValueError is raised.

        Args:
            start_time (int, float):     The time point of the beginning of a period from which to get periods.
            time_unit: (TimeUnit, str):  Time unit for parameter 'start_time'.
            n_periods  (int):            The number of periods to return
            channel_indices (list):      Optional list of channel indices to extract from PSG. Extracts all with None.

        Returns:
            psg: ndarray of shape [n_periods, self.data_per_period, C]
        """
        start_time = self._to_internal_period_time(start_time, time_unit)
        start_idx = self.period_time_to_idx(start_time, self.time_unit)
        return self.get_psg_periods_by_idx(start_idx, n_periods, channel_indices)

    def get_hyp_periods_by_time(self,
                                start_time: [int, float],
                                time_unit: Union[TimeUnit, str],
                                n_periods: int = 1,
                                on_overlapping: Union[str, None] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Equivalent method to self.get_hyp_periods_by_idx but working in time units instead of period indices.

        Returns periods from the hypnogram in shape [n_periods] starting from a time point 'start_time'.
        'start_time' must align exactly with the beginning of a period, otherwise a ValueError is raised.

        Args:
            start_time (int, float):      The time point of the beginning of a period from which to get periods.
            time_unit: (TimeUnit, str):   Time unit for parameter 'start_time'.
            n_periods (int):              The number of periods to return
            on_overlapping (str or None): If str one of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a
                                          discrete period of length self.period_length overlaps 2 or more different
                                          classes in the original hypnogram. See SparseHypnogram.get_period_at_time
                                          for details. Default with on_overlapping = None is self.on_overlapping.

        Returns:
            hyp: ndarray of shape [n_periods]
        """
        start_time = self._to_internal_period_time(start_time, time_unit)
        start_idx = self.period_time_to_idx(start_time, self.time_unit)
        return self.get_hyp_periods_by_idx(start_idx, n_periods, on_overlapping)

    def get_periods_by_time(self,
                            start_time: [int, float],
                            time_unit: Union[TimeUnit, str],
                            n_periods: int = 1,
                            channel_indices: list = None,
                            on_overlapping: str = "RAISE") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Equivalent method to self.get_periods_by_idx but working in time units instead of period indices.

        Get a range of period of {X, y} data by time (self.no_hypnogram is False) else {X} starting from a time
        point 'start_time'. 'start_time' must align exactly with the beginning of a period, otherwise a
        ValueError is raised.

        Args:
            start_time (int, float):     The time point of the beginning of a period from which to get periods.
            time_unit: (TimeUnit, str):  Time unit for parameter 'start_time'.
            n_periods (int):             The number of periods to return
            channel_indices (list):      Optional list of channel indices to extract from PSG. Extracts all with None.
            on_overlapping: (str):       One of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a discrete
                                         period of length self.period_length overlaps 2 or more different classes in the
                                         original hypnogram. See SparseHypnogram.get_period_at_time for details.

        Returns:
            psg: ndarray of shape [n_periods, self.data_per_period, C]
            hyp: ndarray of shape [n_periods] IF self.no_hypnogram is False
        """
        start_time = self._to_internal_period_time(start_time, time_unit)
        start_idx = self.period_time_to_idx(start_time, self.time_unit)
        return self.get_periods_by_idx(start_idx, n_periods, channel_indices, on_overlapping)

    def get_all_psg_periods(self, channel_indices: list = None) -> np.ndarray:
        """
        Returns the full PSG in periods, i.e., shape [self.n_periods, self.data_per_period, n_channels]

        Args:
            channel_indices (list): Optional list of channel indices to extract from PSG. Extracts all with None.
        """
        return self.get_psg_periods_by_idx(0, self.n_periods, channel_indices)

    def get_all_hypnogram_periods(self, on_overlapping: [str, None] = None) -> np.ndarray:
        """
        Returns the full hypnogram in periods, i.e., shape [self.n_periods]

        Args:
            on_overlapping: str, One of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a discrete
                                 period of length self.period_length overlaps 2 or more different classes in the
                                 original hypnogram. See SparseHypnogram.get_period_at_time for details.

        """
        return self.get_hyp_periods_by_idx(0, self.n_periods, on_overlapping)

    def get_all_periods(self,
                        channel_indices: list = None,
                        on_overlapping: [str, None] = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Returns the full (dense) data of the SleepStudy

        Args:
            channel_indices (list):       Optional list of channel indices to extract from PSG. Extracts all with None.
            on_overlapping (str or None): If str one of 'FIRST', 'LAST', 'MAJORITY'. Controls the behaviour when a
                                          discrete period of length self.period_length overlaps 2 or more different
                                          classes in the original hypnogram. See SparseHypnogram.get_period_at_time
                                          for details. Default with on_overlapping = None is self.on_overlapping.

        Returns:
            psg: An ndarray of shape [self.n_periods, self.data_per_period, n_channels]
            hyp: An ndarray of shape [self.n_periods, 1] (if self.no_hypnogram == False)
        """
        psg = self.get_all_psg_periods(channel_indices)
        if self.no_hypnogram:
            return psg
        hyp = self.get_all_hypnogram_periods(on_overlapping)
        self._assert_similar_lengths(psg, hyp)
        return psg, hyp

    @contextmanager
    def loaded_in_context(self, allow_missing_channels=False):
        """ Context manager from automatic loading and unloading """
        self.load(allow_missing_channels=allow_missing_channels)
        try:
            yield self
        finally:
            self.unload()

    @property
    def psg(self):
        """ Returns the PSG object, type depends on concrete implementation """
        return self._psg

    @property
    def hypnogram(self):
        """ Returns the hypnogram (see psg_utils.hypnogram), may be None """
        return self._hypnogram

    @property
    def data_per_period(self) -> int:
        """
        Computes and returns the data (samples) per period of 'period_length' time
        """
        dpp = self.get_period_length_in(TimeUnit.SECOND) * self.sample_rate
        if not math.isclose(dpp, int(dpp)):
            self.raise_err(ValueError, f"Cannot compute data_per_period with period_length {self.period_length} "
                                       f"({self.time_unit}) and sample_rate {self.sample_rate} (Hz) "
                                       f"as the result {dpp} cannot be safely cast to an integer value.")
        return int(dpp)

    def raise_err(self, err_obj, err_msg, _from=None):
        """
        Helper method for raising an error specific to this SleepStudy
        object
        """
        e = err_obj("[{}] {}".format(repr(self), err_msg))
        if _from:
            raise e from _from
        else:
            raise e

    @property
    def _try_channels(self) -> list:
        """ Returns the select and alternative select channels together """
        if len(self.alternative_select_channels[0]) != 0:
            try_channels = [self.select_channels] + self.alternative_select_channels
        else:
            try_channels = [self.select_channels]
        return try_channels

    @property
    def select_channels(self) -> list:
        """ See setter method. """
        return self._select_channels or []

    @select_channels.setter
    def select_channels(self, channels):
        """
        Sets select_channels; a property that when set marks a list of
        channel names to select from the PSG file on disk. All other channels
        are not loaded or removed after loading.

        OBS setting this property when self.loaded is True forces a reload

        Args:
            channels: A list of channel names (strings) giving the names of
                      all channels to load when calling self.load().
        """
        if channels is not None:
            if not isinstance(channels, (list, tuple)):
                self.raise_err(TypeError, f"'channels' must be a list or tuple, got {type(channels)}.")
            if not all([isinstance(c, str) for c in channels]):
                self.raise_err(TypeError, f"Some values in 'select_channels' are not "
                                          f"strings, got {channels}. Expected a flat list of "
                                          f"strings.")
        channels = channels or []
        self._select_channels = channels
        if self.loaded:
            self.reload(warning=True, allow_missing_channels=False)

    @property
    def alternative_select_channels(self) -> list:
        """ See setter method """
        return self._alternative_select_channels or [[]]

    @alternative_select_channels.setter
    def alternative_select_channels(self, channels):
        """
        Set the alternative_select_channels; a property that when set defines
        a list of lists each similar to self.select_channels (see docstring).
        Each define an alternative set of channel names to be loaded in case of
        ChannelNotFound errors in self.load().

        OBS setting this propery when self.loaded is True forces a reload

        Args:
            channels: A list of lists of strings
        """
        e = "'channels' must be a list of lists, where the sub-lists are the "\
            "same lengths as the 'select_channels' list. Got {}."
        if not self.select_channels:
            self.raise_err(ValueError, "Must select primary channels before alternative.")
        if channels is not None:
            if not isinstance(channels, (list, tuple)):
                self.raise_err(TypeError, e.format(channels))
            if len(channels) == 1:
                channels = [ensure_list_or_tuple(channels[0])]
            for chan in channels:
                if not isinstance(chan, (list, tuple)):
                    self.raise_err(TypeError, e.format(type(channels)))
                if len(chan) != len(self.select_channels):
                    self.raise_err(ValueError, e.format(channels))
                if not all([isinstance(c, str) for c in chan]):
                    self.raise_err(TypeError, f"Some values in one of the sub-list of "
                                              f"alternative_select_channels are not "
                                              f"strings, got {channels}. Expected a list of lists"
                                              f" of strings.")
        channels = channels or [[]]
        self._alternative_select_channels = channels
        if self.loaded:
            self.reload(warning=True, allow_missing_channels=False)

    def _set_loaded_channels(self, loaded_channels):
        """
        TODO
        Returns:

        """
        self._select_channels = loaded_channels   # OBS must set private
        self._alternative_select_channels = None  # OBS must set private

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
            X: ndarray of shape [batch_size, self.data_per_period, self.n_channels]
            y: ndarray of shape [batch_size, 1]
        """
        x_batch, y_batch = [], []
        for idx in range(self.n_periods):
            x, y = self.get_periods_by_idx(idx)
            x_batch.append(x), y_batch.append(y)
            if len(x_batch) == batch_size:
                # Note: must copy if overlapping=True
                yield np.array(x_batch), np.array(y_batch)
                if overlapping:
                    x_batch.pop(0), y_batch.pop(0)
                else:
                    x_batch, y_batch = [], []
        if len(x_batch) != 0 and not overlapping:
            yield np.array(x_batch), np.array(y_batch)

    def plot_period(self, period_idx=None, period_sec=None, out_path=None):
        """
        Plot a period of data by index or second

        Args:
            period_idx: Period index to plot
            period_sec: The starting second of the period to plot
            out_path:   Path to save the figure to
        """
        # TODO
        raise NotImplementedError('TODO')

        from psg_utils.visualization.psg_plotting import plot_period
        if bool(period_idx) == bool(period_sec):
            raise ValueError("Must specify either period_idx or period_sec.")
        period_sec = period_sec or self.period_idx_to_sec(period_idx)
        x = self.get_psg_period_at_sec(period_sec)
        if not self.no_hypnogram:
            y = self.get_stage_at_sec(period_sec)
            y = Defaults.get_class_int_to_stage_string()[y]
        else:
            y = None
        plot_period(X=x, y=y,
                    channel_names=self.select_channels,
                    init_second=period_sec,
                    sample_rate=self.sample_rate,
                    out_path=out_path)

    def plot_periods(self, period_idxs=None, period_secs=None, out_path=None,
                     highlight_periods=True):
        """
        Plot multiple periods of data by indices or seconds

        Args:
            period_idxs:        Indices for all periods to plot
            period_secs:        The starting seconds of the periods to plot
            out_path:           Path to save the figure to
            highlight_periods:  Plot period-separating vertical lines
        """
        # TODO
        raise NotImplementedError('TODO')

        from psg_utils.visualization.psg_plotting import plot_periods
        if bool(period_idxs) == bool(period_secs):
            raise ValueError("Must specify either period_idxs or period_secs.")
        period_secs = list(period_secs or map(self.period_idx_to_sec, period_idxs))
        if any(np.diff(period_secs) != self.get_period_length_in(TimeUnit.SECOND)):
            raise ValueError("Periods to plot must be consecutive.")
        xs = list(map(self.get_psg_period_at_sec, period_secs))
        if not self.no_hypnogram:
            ys = list(map(self.get_stage_at_sec, period_secs))
            ys = [Defaults.get_class_int_to_stage_string()[y] for y in ys]
        else:
            ys = None
        plot_periods(X=xs,
                     y=ys,
                     channel_names=self.select_channels,
                     init_second=period_secs[0],
                     sample_rate=self.sample_rate,
                     out_path=out_path,
                     highlight_periods=highlight_periods)
