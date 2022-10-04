import pytest
import numpy as np
from functools import partial
from psg_utils.errors import NotLoadedError
from psg_utils.preprocessing.quality_control_funcs import zero_out_noisy_epochs, clip_noisy_values, apply_quality_control_func
from psg_utils.time_utils import TimeUnit
from tests.dataset.sleep_study.test_sleep_study import standard_study_folder, raw_sleep_study


@pytest.fixture(scope='function')
def noisy_psg_array():
    arr = np.sin(np.linspace(0, np.pi*4, 100))
    # Stack two reversed signals to form two quick channels, second channel higher magnitude
    arr = np.stack([arr, 100 * arr[::-1]]).T  # IQR c1: 1.565, c2: 143.633
    arr[10, 0] = 100
    arr[50:55, 0] = -10
    arr[96:, 0] = -35
    arr[49, 1] = -np.inf
    arr[50, 1] = np.inf
    yield arr


@pytest.fixture(scope='function')
def iqrs(noisy_psg_array):
    yield np.diff(np.percentile(noisy_psg_array, [25, 75], axis=0), axis=0).ravel()


@pytest.fixture(scope='function')
def raw_sleep_study_with_noise(raw_sleep_study):
    raw_sleep_study._psg = raw_sleep_study.psg.copy()
    raw_sleep_study.psg[5, 0] = np.inf
    yield raw_sleep_study


class TestQualityControlFuncs:
    def test_iqr(self, iqrs):
        assert not np.any(np.isnan(iqrs))
        assert not np.any(np.isinf(iqrs))
        assert np.isclose(iqrs[0], 1.565615195156)
        assert np.isclose(iqrs[1], 143.63371114)

    def test_zero_out_noisy_epochs(self, noisy_psg_array):
        psg_copy = noisy_psg_array.copy()
        psg, inds = zero_out_noisy_epochs(noisy_psg_array, sample_rate=2, period_length_sec=5, max_times_global_iqr=20)

        # Check return types and shapes
        assert isinstance(psg, np.ndarray)
        assert isinstance(inds, list)
        assert len(inds) == 2
        assert isinstance(inds[0], np.ndarray)

        # Check that operations are in-place
        assert noisy_psg_array is psg

        # Check correct inds affected
        assert len(inds[0]) == 2
        assert np.all(inds[0] == [1, 9])
        assert len(inds[1]) == 2
        assert np.all(inds[1] == [4, 5])

        # Check specific values were indeed zeroed out
        assert not np.isclose(psg_copy[10:20, 0], 0).any()
        assert np.isclose(psg[10:20, 0], 0).all()
        assert not np.isclose(psg[9:20, 0], 0).all()
        assert not np.isclose(psg[10:21, 0], 0).all()

        # Check that other values were NOT zeroed out
        assert np.isclose(psg_copy[20:90, 0], psg[20:90, 0]).all()

        # Check that INFs were removed
        assert not np.any(np.isinf(psg))

        # Check that no NaNs appeared
        assert not np.any(np.isnan(psg))

        # Check that error is raised with NaN values
        with pytest.raises(ValueError):
            zero_out_noisy_epochs(np.array([[np.nan, 1.0]]), 1, 1)

    def test_clip_noisy_values(self, noisy_psg_array, iqrs):
        psg_copy = noisy_psg_array.copy()
        psg, inds = clip_noisy_values(noisy_psg_array, sample_rate=2, period_length_sec=5, min_max_times_global_iqr=20)
        max_iqrs = 20 * iqrs

        # Check return types and shapes
        assert isinstance(psg, np.ndarray)
        assert isinstance(inds, list)
        assert len(inds) == 2
        assert isinstance(inds[0], np.ndarray)

        # Check that operations are in-place
        assert noisy_psg_array is psg

        # Check correct inds affected
        assert len(inds[0]) == 2
        assert np.all(inds[0] == [1, 9])
        assert len(inds[1]) == 2
        assert np.all(inds[1] == [4, 5])

        # Check specific values were indeed capped to IQR * 20 out
        assert np.isclose(psg[0:10, 0], psg_copy[0:10, 0]).all()
        assert np.isclose(psg[10, 0], max_iqrs[0]).all()
        assert np.isclose(psg[11:96, 0], psg_copy[11:96, 0]).all()
        assert np.isclose(psg[96, 0], -1 * max_iqrs[0]).all()  # OBS: negative 20*IQR
        assert np.isclose(psg[97: 0], psg_copy[97: 0]).all()
        assert np.isclose(psg[49, 1], -1 * max_iqrs[1]).all()  # OBS: negative 20*IQR
        assert np.isclose(psg[50, 1], max_iqrs[1]).all()
        assert np.isclose(psg[:49, 1], psg_copy[:49, 1]).all()
        assert np.isclose(psg[51:, 1], psg_copy[51:, 1]).all()

        # Check that INFs were removed
        assert not np.any(np.isinf(psg))

        # Check that no NaNs appeared
        assert not np.any(np.isnan(psg))

        # Check that error is raised with NaN values
        with pytest.raises(ValueError):
            clip_noisy_values(np.array([[np.nan, 1.0]]), 1, 1)

    def test_apply_quality_control_func_not_loaded(self, raw_sleep_study):
        raw_sleep_study.unload()
        with pytest.raises(NotLoadedError):
            apply_quality_control_func(raw_sleep_study, sample_rate=10, warn=False)

    def test_apply_quality_control_func_not_set(self, raw_sleep_study):
        with pytest.raises(TypeError):
            apply_quality_control_func(raw_sleep_study, sample_rate=10, warn=False)

    def test_apply_quality_control_func_no_noise(self, raw_sleep_study):
        psg_copy = raw_sleep_study.psg.copy()
        # Set QA func (on hidden to not trigger reload and application)
        raw_sleep_study._quality_control_func = ('clip_noisy_values', {"min_max_times_global_iqr": 20})
        psg_r = apply_quality_control_func(raw_sleep_study, sample_rate=10, warn=False)
        assert psg_r is raw_sleep_study.psg
        assert np.all(np.isclose(raw_sleep_study.psg, psg_copy))

    def test_apply_quality_control_func_with_noise_zero(self, raw_sleep_study_with_noise):
        psg_copy = raw_sleep_study_with_noise.psg.copy()

        # Set QA func (on hidden to not trigger reload and application)
        raw_sleep_study_with_noise._quality_control_func = ('zero_out_noisy_epochs', {"max_times_global_iqr": 20})
        psg_r = apply_quality_control_func(raw_sleep_study_with_noise, sample_rate=10, warn=False)
        assert psg_r is raw_sleep_study_with_noise.psg
        assert np.isclose(psg_r[5, 0], 0)

        # Manually apply 'clip_noisy_values' to org PSG array to ensure similar results
        period_length_sec = raw_sleep_study_with_noise.get_period_length_in(TimeUnit.SECOND)
        psg_2, _ = zero_out_noisy_epochs(psg_copy, sample_rate=10, period_length_sec=period_length_sec, max_times_global_iqr=20)
        assert np.all(np.isclose(raw_sleep_study_with_noise.psg, psg_2))

    def test_apply_quality_control_func_with_noise_clip(self, raw_sleep_study_with_noise):
        psg_copy = raw_sleep_study_with_noise.psg.copy()

        # Set QA func (on hidden to not trigger reload and application)
        raw_sleep_study_with_noise._quality_control_func = ('clip_noisy_values', {"min_max_times_global_iqr": 20})
        psg_r = apply_quality_control_func(raw_sleep_study_with_noise, sample_rate=10, warn=False)
        assert psg_r is raw_sleep_study_with_noise.psg
        assert not np.isinf(psg_r[5, 0])

        # Manually apply 'clip_noisy_values' to org PSG array to ensure similar results
        period_length_sec = raw_sleep_study_with_noise.get_period_length_in(TimeUnit.SECOND)
        psg_2, _ = clip_noisy_values(psg_copy, sample_rate=10, period_length_sec=period_length_sec, min_max_times_global_iqr=20)
        assert np.all(np.isclose(raw_sleep_study_with_noise.psg, psg_2))
