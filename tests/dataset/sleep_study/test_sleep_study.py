import pytest
import numpy as np
from psg_utils.dataset.sleep_study import SleepStudy
from psg_utils.io import to_h5_file, to_ids
from psg_utils.time_utils import TimeUnit


@pytest.fixture(scope='module')
def standard_study_folder(tmpdir_factory):
    """
    Yields a temporary dict fixture containing typical PSG and hypnogram (ids) content.
    """
    tmp_path = tmpdir_factory.mktemp('study_folder')
    # Make a PSG (h5) file
    to_h5_file(
        tmp_path / 'psg.h5',
        data=np.arange(315).reshape(105, 3),
        channel_names=['EEG Fpz-Cz', 'C3', 'EOGV'],
        sample_rate=10,
        date=None
    )
    to_ids(
        start=(0, 3, 4.5, 9.0),
        durs=(3, 1.5, 4.5, 1.5),
        stage=['W', 'N1', 'N2', 'REM'],
        out=tmp_path / 'hypnogram.ids'
    )
    # Add some other file to ensure it does not mess up auto-discovery of proper files
    with open(tmp_path / 'some_other_file.ids', 'w') as out_f:
        out_f.write("Some data")
    yield tmp_path


@pytest.fixture(scope='function')
def raw_sleep_study(standard_study_folder):
    study = SleepStudy(
        standard_study_folder,
        psg_regex='psg.h5',
        hyp_regex='hypnogram.ids',
        annotation_dict={'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'UNKNOWN': 5},
        period_length=1.5,
        time_unit=TimeUnit.SECOND,
        internal_time_unit=TimeUnit.MILLISECOND,
        load=False
    )
    study.load()
    yield study


class TestSleepStudy:
    def test_is_loaded(self, raw_sleep_study):
        assert raw_sleep_study.loaded
        assert raw_sleep_study.psg is not None
        assert raw_sleep_study.hypnogram is not None

    def test_is_raw(self, raw_sleep_study):
        assert raw_sleep_study.scaler is None
        assert raw_sleep_study.strip_func is None
        assert raw_sleep_study.quality_control_func is None
        assert len(raw_sleep_study.alternative_select_channels) == 1 and \
               len(raw_sleep_study.alternative_select_channels[0]) == 0
        assert raw_sleep_study.sample_rate == raw_sleep_study.org_sample_rate
        assert len(raw_sleep_study.select_channels) == 3

    def test_sample_rate(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.sample_rate, int)
        assert isinstance(raw_sleep_study.org_sample_rate, int)
        assert raw_sleep_study.sample_rate == 10
        assert raw_sleep_study.org_sample_rate == 10

    def test_period_length(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.period_length, int)
        assert raw_sleep_study.period_length == 1500
        assert raw_sleep_study.get_period_length_in(TimeUnit.SECOND) == 1.5

    def test_recording_length(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.recording_length, int)
        assert raw_sleep_study.recording_length == 10500
        assert isinstance(raw_sleep_study.get_recording_length_in(TimeUnit.SECOND), float)
        assert raw_sleep_study.get_recording_length_in(TimeUnit.SECOND) == 10.5

    def test_last_period_start(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.last_period_start, int)
        assert raw_sleep_study.last_period_start == 9000
        assert isinstance(raw_sleep_study.get_last_period_start_in(TimeUnit.SECOND), float)
        assert raw_sleep_study.get_last_period_start_in(TimeUnit.SECOND) == 9.0

    def test_n_periods(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.n_periods, int)
        assert raw_sleep_study.n_periods == 7

    def test_data_per_period(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.data_per_period, int)
        assert raw_sleep_study.data_per_period == int(1.5 * 10)

    def test_n_classes(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.n_classes, int)
        assert raw_sleep_study.n_classes == 4

    def test_n_channels(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.n_channels, int)
        assert raw_sleep_study.n_channels == 3
        assert raw_sleep_study.n_channels == raw_sleep_study.n_sample_channels

    def test_psg_shape(self, raw_sleep_study):
        shape = raw_sleep_study.get_psg_shape()
        assert isinstance(shape, tuple)
        assert shape[0] == 105 and shape[1] == 3

    def test_class_to_period_dict(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.class_to_period_dict, dict)
        assert len(raw_sleep_study.class_to_period_dict) == 4
        assert isinstance(raw_sleep_study.class_to_period_dict[0], np.ndarray)
        assert np.all(raw_sleep_study.class_to_period_dict[0] == [0, 1])
        assert raw_sleep_study.class_to_period_dict[4] == 6

    def test_classes(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.classes, np.ndarray)
        assert np.all(raw_sleep_study.classes == [0, 1, 2, 4])

    def test_time_units(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.time_unit, TimeUnit)
        assert isinstance(raw_sleep_study.data_time_unit, TimeUnit)
        assert raw_sleep_study.time_unit == TimeUnit.MILLISECOND
        assert raw_sleep_study.data_time_unit == TimeUnit.SECOND

    def test_get_class_counts(self, raw_sleep_study):
        as_dict = raw_sleep_study.get_class_counts(as_dict=True)
        assert isinstance(as_dict, dict)
        assert len(as_dict) == 4
        assert as_dict[0] == 2 and as_dict[1] == 1 and as_dict[2] == 3 and as_dict[4] == 1

        as_array = raw_sleep_study.get_class_counts(as_dict=False)
        assert isinstance(as_array, np.ndarray)
        assert np.all(as_array == [2, 1, 3, 1])

    def test_period_idx_to_time(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.period_idx_to_time(0), int)

        # Test bounds
        with pytest.raises(IndexError):
            raw_sleep_study.period_idx_to_time(-1)
        with pytest.raises(IndexError):
            raw_sleep_study.period_idx_to_time(raw_sleep_study.n_periods)

        assert raw_sleep_study.period_idx_to_time(0) == 0
        assert raw_sleep_study.period_idx_to_time(3) == 4500
        assert raw_sleep_study.period_idx_to_time(raw_sleep_study.n_periods-1) == 9000

    def test_period_time_to_idx(self, raw_sleep_study):
        assert isinstance(raw_sleep_study.period_time_to_idx(0, TimeUnit.SECOND), int)

        # Test bounds
        with pytest.raises(IndexError):
            raw_sleep_study.period_time_to_idx(-1.5, TimeUnit.SECOND)
        with pytest.raises(IndexError):
            raw_sleep_study.period_time_to_idx(raw_sleep_study.recording_length, raw_sleep_study.time_unit)

        # Check non-period alignment
        with pytest.raises(ValueError):
            raw_sleep_study.period_time_to_idx(1.0, TimeUnit.SECOND)

        # Check specific values
        assert raw_sleep_study.period_time_to_idx(1.5, TimeUnit.SECOND) == 1
        assert raw_sleep_study.period_time_to_idx(3.0, TimeUnit.SECOND) == 2
        assert raw_sleep_study.period_time_to_idx(3000, TimeUnit.MILLISECOND) == 2
        assert raw_sleep_study.period_time_to_idx(raw_sleep_study.last_period_start, raw_sleep_study.time_unit) \
               == raw_sleep_study.n_periods - 1

    def test_get_periods_by_idx(self, raw_sleep_study):
        x, y = raw_sleep_study.get_periods_by_idx(0, 1)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert len(y) == len(x) == 1
        assert x.ndim == 3
        assert y.ndim == 1
        assert x.shape[1] == raw_sleep_study.data_per_period
        assert x.shape[2] == raw_sleep_study.n_channels
        assert y == 0

        # Assert x is a view of the raw PSG array
        assert x.base is raw_sleep_study.psg

        # Assert some values
        assert np.all(np.isclose(x, raw_sleep_study.psg[:raw_sleep_study.data_per_period]))

        # Check channel indices
        x_chan, _ = raw_sleep_study.get_periods_by_idx(0, 1, channel_indices=[1])
        assert x_chan.ndim == 3
        assert x_chan.shape[-1] == 1
        assert np.all(np.isclose(x_chan.ravel(), x[..., 1].ravel()))

        # Check multiple periods
        with pytest.raises(IndexError):
            raw_sleep_study.get_periods_by_idx(0, -1)
        with pytest.raises(IndexError):
            raw_sleep_study.get_periods_by_idx(0, raw_sleep_study.n_periods + 1)

        x, y = raw_sleep_study.get_periods_by_idx(2, 3)
        assert len(x) == len(y) == 3
        assert np.all(y == [1, 2, 2])
        assert x.base is raw_sleep_study.psg
        assert np.all(
            np.isclose(x.reshape(-1, raw_sleep_study.n_channels),
                       raw_sleep_study.psg[raw_sleep_study.data_per_period * 2:
                                           raw_sleep_study.data_per_period * (2 + 3)]
                       )
        )

        # Check if only returning 1 PSG if no_hypnogam is set
        raw_sleep_study._no_hypnogram = True
        x1 = raw_sleep_study.get_periods_by_idx(2, 3)
        assert isinstance(x1, np.ndarray)
        assert np.all(np.isclose(x, x1))

    def test_get_psg_periods_by_time(self, raw_sleep_study):
        x, y = raw_sleep_study.get_periods_by_time(0, TimeUnit.SECOND, 1)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert len(y) == len(x) == 1
        assert x.ndim == 3
        assert y.ndim == 1
        assert x.shape[1] == raw_sleep_study.data_per_period
        assert x.shape[2] == raw_sleep_study.n_channels
        assert y == 0

        with pytest.raises(ValueError):
            # Not on period start
            raw_sleep_study.get_periods_by_time(1.0, TimeUnit.SECOND)
        with pytest.raises(IndexError):
            # Negative time index
            raw_sleep_study.get_periods_by_time(-1500, TimeUnit.MILLISECOND)
        with pytest.raises(IndexError):
            # Out of bounds index
            raw_sleep_study.get_periods_by_time(raw_sleep_study.recording_length, TimeUnit.MILLISECOND)

        # Check matches against index version
        x1, y1 = raw_sleep_study.get_periods_by_time(raw_sleep_study.period_length * 2, raw_sleep_study.time_unit, 3)
        x2, y2 = raw_sleep_study.get_periods_by_idx(2, 3)
        assert np.all(np.isclose(x1, x2))
        assert np.all(y1 == y2)
