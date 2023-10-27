import pytest
import numpy as np
from psg_utils.preprocessing.psg_sampling import set_psg_sample_rate, poly_resample, fourier_resample


@pytest.fixture(scope='function')
def psg_array():
    return np.arange(200).reshape([100, 2]).astype(np.float32)


class TestPSGSampling:
    def test_poly_resample(self, psg_array):
        psg_resampled = poly_resample(psg_array, new_sample_rate=1, old_sample_rate=2)
        assert psg_resampled.shape[0] == 50
        assert psg_resampled.shape[1] == 2

        # Assert output type, should be float64 but if other versions cast to float32 that is fine as well
        assert psg_resampled.dtype in (np.float64, np.float32)
        assert poly_resample(psg_array.astype(int), new_sample_rate=1, old_sample_rate=2).dtype in (np.float64, np.float32)

        # Test on other datatypes, should not raise an error and should give similar results
        assert np.all(np.isclose(
            poly_resample(psg_array.astype(np.float32), new_sample_rate=1, old_sample_rate=2),
            psg_resampled
        ))
        # Should be similar values even when cast to ints as input is integers converted to float64
        assert np.all(np.isclose(
            poly_resample(psg_array.astype(int), new_sample_rate=1, old_sample_rate=2),
            psg_resampled
        ))

    def test_fourier_resample(self, psg_array):
        psg_resampled = fourier_resample(psg_array, new_sample_rate=1, old_sample_rate=2)
        assert psg_resampled.shape[0] == 50
        assert psg_resampled.shape[1] == 2

        # Assert output type, should be float64 but if other versions cast to float32 that is fine as well
        assert psg_resampled.dtype in (np.float64, np.float32)
        assert fourier_resample(psg_array.astype(int), new_sample_rate=1, old_sample_rate=2).dtype in (np.float64, np.float32)

        # Test on other datatypes, should not raise an error and should give similar results
        assert np.all(np.isclose(
            fourier_resample(psg_array.astype(np.float32), new_sample_rate=1, old_sample_rate=2),
            psg_resampled
        ))
        # Should be similar values even when cast to ints as input is integers converted to float64
        assert np.all(np.isclose(
            fourier_resample(psg_array.astype(int), new_sample_rate=1, old_sample_rate=2),
            psg_resampled
        ))

    def test_poly_vs_fourier(self, psg_array):
        # Chefk that the two methods produce different results
        psg_1 = poly_resample(psg_array, new_sample_rate=1, old_sample_rate=2)
        psg_2 = fourier_resample(psg_array, new_sample_rate=1, old_sample_rate=2)
        assert not np.all(np.isclose(psg_1, psg_2))

    def test_set_psg_sample_rate(self, psg_array):
        # Test poly
        psg_poly_1 = set_psg_sample_rate(psg_array, new_sample_rate=1, old_sample_rate=2, method='poly')
        psg_poly_2 = poly_resample(psg_array, new_sample_rate=1, old_sample_rate=2)
        assert np.all(np.isclose(psg_poly_1, psg_poly_2))

        # Test fourier
        psg_fourier_1 = set_psg_sample_rate(psg_array, new_sample_rate=1, old_sample_rate=2, method='fourier')
        psg_fourier_2 = fourier_resample(psg_array, new_sample_rate=1, old_sample_rate=2)
        assert np.all(np.isclose(psg_fourier_1, psg_fourier_2))

        # Test raises with unknown resampling type
        with pytest.raises(ValueError):
            set_psg_sample_rate(psg_array, new_sample_rate=1, old_sample_rate=2, method='my_sampler')
