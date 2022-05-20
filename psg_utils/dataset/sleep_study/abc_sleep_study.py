import logging
import numpy as np
import math
from typing import Tuple, Union
from abc import ABC, abstractmethod
from contextlib import contextmanager
from psg_utils import Defaults
from psg_utils.utils import ensure_list_or_tuple


logger = logging.getLogger(__name__)


class AbstractBaseSleepStudy(ABC):
    """
    TODO
    """
    def __init__(self,
                 annotation_dict,
                 period_length_sec,
                 no_hypnogram):
        """
        TODO

        Args:
            annotation_dict:
            period_length_sec:
            no_hypnogram:
        """
        self.annotation_dict = annotation_dict
        self.no_hypnogram = no_hypnogram
        self.period_length_sec = period_length_sec or Defaults.get_default_period_length()

        # Hidden attributes controlled in property functions to limit setting
        # of these values to the load() function
        self._psg = None
        self._hypnogram = None
        self._class_to_period_dict = None
        self._select_channels = None
        self._alternative_select_channels = None

    @abstractmethod
    def identifier(self):
        raise NotImplemented

    @abstractmethod
    def __str__(self):
        raise NotImplemented

    @abstractmethod
    def __repr__(self):
        raise NotImplemented

    @abstractmethod
    def loaded(self):
        raise NotImplemented

    @abstractmethod
    def reload(self, warning, allow_missing_channels):
        raise NotImplemented

    @abstractmethod
    def load(self, allow_missing_channels):
        raise NotImplemented

    @abstractmethod
    def unload(self):
        raise NotImplemented

    @abstractmethod
    def get_psg_shape(self) -> tuple:
        raise NotImplemented

    @abstractmethod
    def get_full_psg(self) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def get_full_hypnogram(self, on_overlapping: str = "RAISE") -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def get_all_periods(self, on_overlapping: str = "RAISE") -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        raise NotImplemented

    @abstractmethod
    def get_psg_period_at_sec(self, second: [int, float]) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def get_stage_at_sec(self, second: [int, float]) -> Union[int, np.ndarray]:
        raise NotImplemented

    @abstractmethod
    def get_class_indicies(self, class_int):
        raise NotImplemented

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        raise NotImplemented

    @property
    @abstractmethod
    def date(self):
        raise NotImplemented

    @property
    @abstractmethod
    def n_classes(self) -> int:
        raise NotImplemented

    @property
    @abstractmethod
    def recording_length_sec(self) -> float:
        raise NotImplemented

    @abstractmethod
    def extract_from_psg(self, start_second: Union[float, int], end_second: Union[float, int], channel_inds: list = None):
        """
        Extract PSG data from second 'start_second' (inclusive) to second 'end_second'
        (exclusive)

        Args:
            start_second: int or float, start_second second to extract from
            end_second: int or flaot, end_second second to extract from
            channel_inds: list, list of channel indices to extract from

        Returns:
            A numpy view of self.psg
        """
        raise NotImplemented

    @property
    def last_period_start_sec(self) -> float:
        """
        Returns the second that marks the beginning of the last period
        """
        reminder = self.recording_length_sec % self.period_length_sec
        if reminder > 0:
            return self.recording_length_sec - reminder
        else:
            return self.recording_length_sec - self.period_length_sec

    @property
    def n_periods(self) -> int:
        """
        Returns the number of periods of length self.period_length in the PSG.
        Note that this include any partial final period of less than self.period_length_sec length.
        """
        return int(np.ceil(self.recording_length_sec / self.period_length_sec))

    @property
    def n_channels(self) -> int:
        """ Returns the number of channels in the PSG array """
        return len(self.select_channels)

    @property
    def n_sample_channels(self) -> int:
        """
        Overwritten in some derived classes that sample channels
        on-access
        """
        return self.n_channels

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
        Computes and returns the data (samples) per period of
        'period_length' seconds of time (en 'epoch' in sleep research)
        """
        dpp = self.period_length_sec * self.sample_rate
        if not math.isclose(dpp, int(dpp)):
            raise ValueError(f"Cannot compute data_per_period with period_length_sec "
                             f"{self.period_length_sec} and sample_rate {self.sample_rate} as the "
                             f"result {dpp} cannot be safely cast to an integer value.")
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

        OBS setting this propery when self.loaded is True forces a reload

        Args:
            channels: A list of channel names (strings) giving the names of
                      all channels to load when calling self.load().
        """
        if channels is not None:
            if not isinstance(channels, (list, tuple)):
                raise TypeError("'channels' must be a list or tuple, got "
                                "{}.".format(type(channels)))
            if not all([isinstance(c, str) for c in channels]):
                raise TypeError("Some values in 'select_channels' are not "
                                "strings, got {}. Expected a flat list of "
                                "strings.".format(channels))
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
            raise ValueError("Must select primary channels before "
                             "alternative.")
        if channels is not None:
            if not isinstance(channels, (list, tuple)):
                raise TypeError(e.format(channels))
            if len(channels) == 1:
                channels = [ensure_list_or_tuple(channels[0])]
            for chan in channels:
                if not isinstance(chan, (list, tuple)):
                    raise TypeError(e.format(type(channels)))
                if len(chan) != len(self.select_channels):
                    raise ValueError(e.format(channels))
                if not all([isinstance(c, str) for c in chan]):
                    raise TypeError("Some values in one of the sub-list of "
                                    "alternative_select_channels are not "
                                    "strings, got {}. Expected a list of lists"
                                    " of strings.".format(channels))
        channels = channels or [[]]
        self._alternative_select_channels = channels
        if self.loaded:
            self.reload(warning=True, allow_missing_channels=False)

    def period_idx_to_sec(self, period_idx: int) -> [int, float]:
        """
        Helper method that maps a period_idx (int) to the first second in that
        period.
        """
        return period_idx * self.period_length_sec

    def _set_loaded_channels(self, loaded_channels):
        """
        TODO
        Returns:

        """
        self._select_channels = loaded_channels   # OBS must set private
        self._alternative_select_channels = None  # OBS must set private

    def get_period_at_sec(self, second: [int, float]) -> Tuple[np.ndarray, Union[int, np.ndarray]]:
        """
        Get a period of {X, y} data starting at 'second' seconds.

        Returns:
            X: An ndarray of shape [self.data_per_period, self.n_channels]
            y: An ndarray of shape [1]
        """
        x = self.get_psg_period_at_sec(second)
        y = self.get_stage_at_sec(second)
        return x, y

    def get_period_by_idx(self, period_idx: int) -> Tuple[np.ndarray, Union[int, np.ndarray]]:
        """
        Get a period of {X, y} data by index
        Period starting at second 0 is index 0.

        Returns:
            X: An ndarray of shape [self.data_per_period, self.n_channels]
            y: An ndarray of shape [1]
        """
        period_start_sec = self.period_idx_to_sec(period_idx)
        return self.get_period_at_sec(period_start_sec)

    @abstractmethod
    def get_periods_by_idx(self, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a range of period of {X, y} data by indices
        Period starting at second 0 is index 0.

        Args:
            start_idx (int): Index of first period to return
            end_idx   (int): Index of last period to return (inclusive)

        Returns:
            X: ndarray of shape [N periods, self.data_per_period, C]
            y: ndarray of shape [N periods, 1]
        """
        raise NotImplemented

    def get_stage_by_idx(self, period_idx: int) -> Union[np.ndarray, int]:
        """
        Get the hypnogram stage at period index 'period_idx'.

        Returns:
            y: An ndarray of shape [1]
        """
        period_start_sec = self.period_idx_to_sec(period_idx)
        return self.get_stage_at_sec(period_start_sec)

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
        x_batch, y_batch = [], []
        for idx in range(self.n_periods):
            x, y = self.get_period_by_idx(idx)
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
        from psg_utils.visualization.psg_plotting import plot_periods
        if bool(period_idxs) == bool(period_secs):
            raise ValueError("Must specify either period_idxs or period_secs.")
        period_secs = list(period_secs or map(self.period_idx_to_sec,
                                              period_idxs))
        if any(np.diff(period_secs) != self.period_length_sec):
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
