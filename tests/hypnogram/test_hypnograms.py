import pytest
import numpy as np
from psg_utils.hypnogram import SparseHypnogram, DenseHypnogram
from psg_utils.time_utils import TimeUnit


@pytest.fixture(scope='session')
def ids_tuple():
    yield (0, 2, 5, 9), (2, 3, 4, 1), [0, 3, 1, 4]


@pytest.fixture(scope='function')
def hyp(ids_tuple):
    yield SparseHypnogram(
        *ids_tuple,
        period_length=1.5,
        time_unit=TimeUnit.SECOND,
        internal_time_unit=TimeUnit.MILLISECOND
    )


@pytest.fixture(scope='function')
def dense_hyp():
    yield DenseHypnogram(
        dense_init_times=[1.0, 2.0, 3.0],
        dense_stages=[0, 3, 1],
        time_unit=TimeUnit.SECOND
    )


class TestDenseHypnogram:
    def test_column_names(self, dense_hyp):
        assert np.all(dense_hyp.columns == ['period_init_time', 'sleep_stage'])

    def test_init_times(self, dense_hyp):
        assert np.all(dense_hyp['period_init_time'].values == [1.0, 2.0, 3.0])

    def test_stages(self, dense_hyp):
        assert np.all(dense_hyp['sleep_stage'].values == [0, 3, 1])

    def test_shape(self, dense_hyp):
        assert dense_hyp.shape[0] == 3 and dense_hyp.shape[1] == 2

    def test_init_hyp_with_gaps(self):
        with pytest.raises(ValueError):
            DenseHypnogram(
                dense_init_times=[5.0, 6.0, 10.0],
                dense_stages=[0, 3, 1],
                time_unit=TimeUnit.SECOND
            )

    def test_init_hyp_with_negative_inits(self):
        with pytest.raises(ValueError):
            DenseHypnogram(
                dense_init_times=[-1.0, 0.0, 1.0],
                dense_stages=[0, 3, 1],
                time_unit=TimeUnit.SECOND
            )


class TestSparseHypnogram:
    def test_n_classes(self, hyp):
        assert isinstance(hyp.n_classes, int)
        assert hyp.n_classes == 4

    def test_classes(self, hyp):
        assert isinstance(hyp.classes, np.ndarray)
        assert np.all(hyp.classes == [0, 1, 3, 4])

    def test_n_periods(self, hyp):
        assert isinstance(hyp.n_periods, int)
        assert hyp.n_periods == 7

    def test_period_length(self, hyp):
        assert isinstance(hyp.period_length, int)
        assert hyp.period_length == 1500

    def test_period_length_sec(self, hyp):
        assert isinstance(hyp.period_length_sec, float)
        assert hyp.period_length_sec == 1.5

    def test_end_time(self, hyp):
        assert isinstance(hyp.end_time, int)
        assert hyp.end_time == 10000

    def test_end_time_sec(self, hyp):
        assert isinstance(hyp.end_time_sec, float)
        assert hyp.end_time_sec == 10.0

    def test_last_period_start(self, hyp):
        assert isinstance(hyp.last_period_start, int)
        assert hyp.last_period_start == 9000

    def test_last_period_start_sec(self, hyp):
        assert isinstance(hyp.last_period_start_sec, float)
        assert hyp.last_period_start_sec == 9.0

    def test_total_duration(self, hyp):
        assert isinstance(hyp.total_duration, int)
        assert hyp.total_duration == 10000

    def test_total_duration_sec(self, hyp):
        assert isinstance(hyp.total_duration_sec, float)
        assert hyp.total_duration_sec == 10.0

    def test_set_new_end_time_shorter(self, hyp):
        time_unit = TimeUnit.SECOND
        hyp.set_new_end_time(hyp.end_time_sec - 3, time_unit)
        assert hyp.end_time == 7000
        assert hyp.end_time_sec == 7.0
        assert hyp.n_periods == 5
        assert hyp.n_classes == 3
        assert hyp.stages[-1] == 1
        assert hyp.durations[-1] == 2000
        assert hyp.inits[-1] == 5000

    def test_set_new_end_time_longer(self, hyp):
        time_unit = TimeUnit.MILLISECOND
        hyp.set_new_end_time(hyp.end_time + 3000, time_unit)
        assert hyp.end_time == 13000
        assert hyp.end_time_sec == 13.0
        assert hyp.n_periods == 9
        assert hyp.n_classes == 5
        assert hyp.stages[-1] == 5  # UNKNOWN stage
        assert hyp.durations[-1] == 3000

    def test_get_index_at_time(self, hyp):
        assert isinstance(hyp.get_index_at_time(0, TimeUnit.SECOND), int)
        assert hyp.get_index_at_time(1500, TimeUnit.MILLISECOND) == hyp.get_index_at_time(1.5, TimeUnit.SECOND)
        assert hyp.get_index_at_time(1.5, TimeUnit.SECOND) == 0
        assert hyp.get_index_at_time(2.0, TimeUnit.SECOND) == 1
        assert hyp.get_index_at_time(2000, TimeUnit.MILLISECOND) == 1
        assert hyp.get_index_at_time(1999, TimeUnit.MILLISECOND) == 0
        with pytest.raises(IndexError):
            hyp.get_index_at_time(15, TimeUnit.SECOND)  # Out of bounds

    def test_get_period_at_time(self, hyp):
        assert isinstance(hyp.get_period_at_time(0, TimeUnit.SECOND), int)
        assert hyp.get_period_at_time(0, TimeUnit.SECOND) == 0

        # Test bounds
        with pytest.raises(IndexError):
            hyp.get_period_at_time(-1, TimeUnit.SECOND)
        with pytest.raises(IndexError):
            hyp.get_period_at_time(hyp.last_period_start_sec+hyp.period_length_sec, TimeUnit.SECOND)

        # Test in seconds
        with pytest.raises(ValueError):
            hyp.get_period_at_time(1.5, TimeUnit.SECOND, on_overlapping='RAISES')
        assert hyp.get_period_at_time(1.5, TimeUnit.SECOND, on_overlapping='FIRST') == 0
        assert hyp.get_period_at_time(1.5, TimeUnit.SECOND, on_overlapping='LAST') == 3
        assert hyp.get_period_at_time(1.5, TimeUnit.SECOND, on_overlapping='MAJORITY') == 3

        # Test in miliseconds
        with pytest.raises(ValueError):
            hyp.get_period_at_time(4499, TimeUnit.MILLISECOND, on_overlapping='RAISES')
        assert hyp.get_period_at_time(4499, TimeUnit.MILLISECOND, on_overlapping='FIRST') == 3
        assert hyp.get_period_at_time(4499, TimeUnit.MILLISECOND, on_overlapping='LAST') == 3
        assert hyp.get_period_at_time(4500, TimeUnit.MILLISECOND, on_overlapping='LAST') == 1

    def test_get_stage_at_time(self, hyp):
        assert isinstance(hyp.get_stage_at_time(0, TimeUnit.SECOND), int)

        # Test bounds
        with pytest.raises(IndexError):
            hyp.get_stage_at_time(-1, TimeUnit.SECOND)
        with pytest.raises(IndexError):
            hyp.get_stage_at_time(hyp.end_time_sec, TimeUnit.SECOND)

        # Specific tests
        assert hyp.get_stage_at_time(0, TimeUnit.MILLISECOND) == 0
        assert hyp.get_stage_at_time(4999, TimeUnit.MILLISECOND) == 3
        assert hyp.get_stage_at_time(5, TimeUnit.SECOND) == 1
        assert hyp.get_stage_at_time(9999, TimeUnit.MILLISECOND) == 4

    def test_get_class_durations(self, hyp):
        class_durs = hyp.get_class_durations()
        assert isinstance(class_durs, dict)
        assert len(class_durs) == hyp.n_classes
        assert class_durs[0] == 2000
        assert class_durs[1] == 4000
        assert class_durs[3] == 3000
        assert class_durs[4] == 1000
        assert class_durs[2] == 0

    def test_to_dense(self, hyp):
        assert isinstance(hyp.to_dense(on_overlapping='FIRST'), DenseHypnogram)

        with pytest.raises(ValueError):
            hyp.to_dense(on_overlapping="RAISE")

        # Majority in seconds
        dense = hyp.to_dense(on_overlapping='MAJORITY', dense_time_unit=TimeUnit.SECOND)
        assert np.all(dense.columns == ['period_init_time', 'sleep_stage'])
        assert dense.loc[0, 'period_init_time'] == 0
        assert dense.loc[1, 'period_init_time'] == 1.5
        assert dense.loc[len(dense)-1, 'period_init_time'] == hyp.last_period_start_sec
        assert dense.shape[0] == hyp.n_periods
        assert dense.shape[1] == 2
        assert np.all(dense['period_init_time'].values == [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0])
        assert np.all(dense['sleep_stage'].values == [0, 3, 3, 1, 1, 1, 4])

        # FIRST in microseconds
        dense = hyp.to_dense(on_overlapping='FIRST', dense_time_unit=TimeUnit.MICROSECOND)
        assert np.all(dense['period_init_time'].values == [0, 1500000, 3000000, 4500000, 6000000, 7500000, 9000000])
        assert np.all(dense['sleep_stage'].values == [0, 0, 3, 3, 1, 1, 4])

    def test_is_compact(self, hyp):
        assert hyp.is_compact
        hyp.stages[1] = hyp.stages[0]
        assert not hyp.is_compact

    def test_make_compact(self, hyp):
        assert hyp.is_compact
        hyp.stages[1] = hyp.stages[0]
        hyp.make_compact()
        assert hyp.is_compact
        assert np.all(hyp.stages == [0, 1, 4])
        assert np.all(hyp.inits == [0, 5000, 9000])
        assert np.all(hyp.durations == [5000, 4000, 1000])

        with pytest.raises(ValueError):
            # Should raise ValueError on gaps
            hyp.stages[1] = hyp.stages[0]  # make non compact again (otherwise make_compact has no effect)
            hyp.inits[1] += 1  # insert gap
            hyp.make_compact()
