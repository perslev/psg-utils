import pytest
import numpy as np
from psg_utils.preprocessing.strip_funcs import strip_to_match, assert_equal_length
from psg_utils.dataset.sleep_study import SleepStudy
from psg_utils.time_utils import TimeUnit
from tests.dataset.sleep_study.test_sleep_study import standard_study_folder, raw_sleep_study


@pytest.fixture(scope="function")
def sleep_study_hyp_longer(standard_study_folder):
    study_hyp_longer = SleepStudy(
        standard_study_folder,
        psg_regex='psg.h5',
        hyp_regex='hypnogram.ids',
        annotation_dict={'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'UNKNOWN': 5},
        period_length=1.5,
        time_unit=TimeUnit.SECOND,
        internal_time_unit=TimeUnit.MILLISECOND,
        load=False
    )
    study_hyp_longer.load()
    study_hyp_longer.hypnogram.set_new_end_time(12.5, TimeUnit.SECOND)  # org length 10.5 seconds
    yield study_hyp_longer


@pytest.fixture(scope="function")
def sleep_study_hyp_shorter(standard_study_folder):
    study_hyp_shorter = SleepStudy(
        standard_study_folder,
        psg_regex='psg.h5',
        hyp_regex='hypnogram.ids',
        annotation_dict={'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'UNKNOWN': 5},
        period_length=1.5,
        time_unit=TimeUnit.SECOND,
        internal_time_unit=TimeUnit.MILLISECOND,
        load=False
    )
    study_hyp_shorter.load()
    study_hyp_shorter.hypnogram.set_new_end_time(8.5, TimeUnit.SECOND)  # org length 10.5 seconds
    yield study_hyp_shorter


@pytest.fixture(scope="function")
def sleep_study_non_periodic(standard_study_folder):
    # Set period length to 0.4 which is non-periodic for total data length of 10.5 seconds.
    study = SleepStudy(
        standard_study_folder,
        psg_regex='psg.h5',
        hyp_regex='hypnogram.ids',
        annotation_dict={'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'UNKNOWN': 5},
        period_length=0.4,
        time_unit=TimeUnit.SECOND,
        internal_time_unit=TimeUnit.MILLISECOND,
        on_overlapping="MAJORITY",
        load=False
    )
    study.load()
    # Similarly set hypnogram to non-period (and non-matching to PSG) length
    study.hypnogram.set_new_end_time(8.5, TimeUnit.SECOND)  # org length 10.5 seconds
    yield study


class TestStripFuncs:
    def test_fixtures(self, sleep_study_hyp_longer, sleep_study_hyp_shorter):
        assert sleep_study_hyp_longer.hypnogram.total_duration > sleep_study_hyp_longer.recording_length
        assert sleep_study_hyp_shorter.hypnogram.total_duration < sleep_study_hyp_shorter.recording_length

    def test_assert_equal_length(self, sleep_study_hyp_longer, sleep_study_hyp_shorter, raw_sleep_study):
        sr = sleep_study_hyp_shorter.sample_rate
        assert not assert_equal_length(sleep_study_hyp_longer.psg, sleep_study_hyp_longer.hypnogram, sr)
        assert not assert_equal_length(sleep_study_hyp_shorter.psg, sleep_study_hyp_shorter.hypnogram, sr)
        assert assert_equal_length(raw_sleep_study.psg, raw_sleep_study.hypnogram, raw_sleep_study.sample_rate)

    def test_strip_to_match_hyp_longer(self, sleep_study_hyp_longer):
        # Check UNKNWON stage was inserted
        assert sleep_study_hyp_longer.hypnogram.n_classes == 5
        assert np.all(sleep_study_hyp_longer.hypnogram.classes == [0, 1, 2, 4, 5])
        assert sleep_study_hyp_longer.hypnogram.total_duration_sec == 12.5

        # Apply strip to match
        psg, hyp = strip_to_match(
            psg=sleep_study_hyp_longer.psg,
            hyp=sleep_study_hyp_longer.hypnogram,
            sample_rate=sleep_study_hyp_longer.sample_rate,
        )

        # Check now same length
        assert assert_equal_length(psg, hyp, sleep_study_hyp_longer.sample_rate)

        # Check that PSG was not stripped
        assert len(psg) == len(sleep_study_hyp_longer.psg)

        # Check that HYP was made shorter in end
        assert hyp.n_classes == 4
        assert np.all(hyp.classes == [0, 1, 2, 4])
        assert hyp.total_duration_sec == 10.5
        assert hyp.inits[-1] == 9000
        assert hyp.durations[-1] == 1500

    def test_strip_to_match_hyp_shorter(self, sleep_study_hyp_shorter):
        # Check last stage was removed
        assert sleep_study_hyp_shorter.hypnogram.n_classes == 3
        assert np.all(sleep_study_hyp_shorter.hypnogram.classes == [0, 1, 2])
        assert sleep_study_hyp_shorter.hypnogram.total_duration_sec == 8.5

        # Apply strip to match
        psg, hyp = strip_to_match(
            psg=sleep_study_hyp_shorter.psg,
            hyp=sleep_study_hyp_shorter.hypnogram,
            sample_rate=sleep_study_hyp_shorter.sample_rate,
        )

        # Check now same length
        assert assert_equal_length(psg, hyp, sleep_study_hyp_shorter.sample_rate)

        # Check that PSG was not stripped
        assert len(psg) == len(sleep_study_hyp_shorter.psg)

        # # Check that HYP was made longer in end by insering UNKNOWN stage
        assert hyp.n_classes == 4
        assert np.all(hyp.classes == [0, 1, 2, 5])
        assert hyp.total_duration_sec == 10.5
        assert hyp.inits[-1] == 8500
        assert hyp.durations[-1] == 2000
        assert hyp.stages[-1] == 5

    def test_strip_to_match_non_periodic(self, sleep_study_non_periodic):
        # Should not be possible to extract periods due to the PSG length being non-periodic
        with pytest.raises(ValueError):
            sleep_study_non_periodic.get_all_periods(on_overlapping='MAJORITY')

        # Strip PSG and HYP to similar and period-divisible lengths
        psg, hyp = strip_to_match(
            psg=sleep_study_non_periodic.psg,
            hyp=sleep_study_non_periodic.hypnogram,
            sample_rate=sleep_study_non_periodic.sample_rate,
        )

        # PSG length before: 10.5 seconds
        # Period length sec: 0.5 seconds
        # Expected length after: 10.4 (10.5 - (10.5 % 0.4))
        assert len(psg) / sleep_study_non_periodic.sample_rate == 10.4
        assert np.all(np.isclose(psg, sleep_study_non_periodic.psg[:104]))

        # Check equal length to hyp
        assert assert_equal_length(psg, hyp, sleep_study_non_periodic.sample_rate)

        # Hypnogram should have gotton longer by inserting UNKNWON / int 5 stage
        assert hyp.total_duration_sec == 10.4
        assert hyp.stages[-1] == 5
        assert hyp.inits[-1] == 8500
        assert hyp.durations[-1] == 1900

        # It should now be possible to get all periods in MAJORITY mode
        sleep_study_non_periodic._psg = psg
        sleep_study_non_periodic._hypnogram = hyp
        sleep_study_non_periodic.get_all_periods(on_overlapping='MAJORITY')
