"""
Radiometry tests.

The ENL estimator is validated against simulated speckle with a known look
count, which is the only way to be confident it is measuring what it claims.
"""

import numpy as np
import pytest

from sarpanel import radiometry as rad


@pytest.mark.parametrize("true_looks", [1.0, 4.4, 12.0, 40.0])
def test_enl_recovers_known_look_count(true_looks):
    """
    L-look intensity is Gamma(shape=L, scale=sigma/L). The estimator must
    return L to within sampling error, which at n=200000 is well under 1%.
    """
    rng = np.random.default_rng(42)
    sigma = 0.3
    sample = rng.gamma(shape=true_looks, scale=sigma / true_looks, size=200_000)
    assert rad.estimate_enl(sample) == pytest.approx(true_looks, rel=0.02)


@pytest.mark.parametrize("true_looks", [1.0, 4.4, 12.0])
def test_cv_matches_inverse_sqrt_looks(true_looks):
    rng = np.random.default_rng(7)
    sample = rng.gamma(shape=true_looks, scale=1.0 / true_looks, size=200_000)
    assert rad.coefficient_of_variation(sample) == pytest.approx(
        1.0 / np.sqrt(true_looks), rel=0.02
    )


def test_enl_computed_on_db_values_is_wrong():
    """
    Guards the documented failure mode. Estimating ENL on decibel values gives
    a number with no relationship to the look count, because the log transform
    stabilises variance. The estimator must be given linear power.
    """
    rng = np.random.default_rng(1)
    sample = rng.gamma(shape=4.4, scale=1.0 / 4.4, size=100_000)
    correct = rad.estimate_enl(sample)
    wrong = rad.estimate_enl(rad.to_db(sample), already_db=False)
    assert correct == pytest.approx(4.4, rel=0.05)
    assert not (3.0 < wrong < 6.0)


def test_already_db_flag_round_trips():
    rng = np.random.default_rng(3)
    sample = rng.gamma(shape=8.0, scale=1.0 / 8.0, size=50_000)
    from_linear = rad.estimate_enl(sample)
    from_db = rad.estimate_enl(rad.to_db(sample), already_db=True)
    assert from_db == pytest.approx(from_linear, rel=1e-6)


def test_enl_returns_nan_below_minimum_sample():
    assert np.isnan(rad.estimate_enl(np.array([0.1, 0.2, 0.3])))


def test_db_averaging_is_biased_low():
    """
    The documented reason aggregation happens in linear power. The bias grows
    with heterogeneity, so it is correlated with within-field variability
    rather than being independent noise.
    """
    rng = np.random.default_rng(11)
    homogeneous = rng.gamma(shape=40.0, scale=1.0 / 40.0, size=100_000)
    heterogeneous = rng.gamma(shape=1.0, scale=1.0, size=100_000)

    for sample in (homogeneous, heterogeneous):
        correct = rad.mean_linear_then_db(sample)
        naive = float(np.nanmean(rad.to_db(sample)))
        assert naive < correct

    bias_low = rad.mean_linear_then_db(homogeneous) - np.nanmean(rad.to_db(homogeneous))
    bias_high = rad.mean_linear_then_db(heterogeneous) - np.nanmean(
        rad.to_db(heterogeneous)
    )
    assert bias_high > bias_low > 0


def test_to_db_maps_nonpositive_to_nan():
    out = rad.to_db(np.array([1.0, 0.0, -1.0, 10.0]))
    assert out[0] == pytest.approx(0.0)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out[3] == pytest.approx(10.0)


def test_standard_error_shrinks_with_pixels_and_looks():
    big = rad.speckle_standard_error_db(1000, 4.4)
    small = rad.speckle_standard_error_db(20, 4.4)
    assert small > big
    # Quadrupling the sample halves the error.
    assert rad.speckle_standard_error_db(400, 4.4) == pytest.approx(
        rad.speckle_standard_error_db(100, 4.4) / 2, rel=1e-6
    )


def test_nominal_looks_slc_versus_grd():
    slc = rad.nominal_looks(10.0, "SLC")
    grd = rad.nominal_looks(10.0, "GRD")
    # GRD arrives already multilooked, so at native spacing it carries the
    # product's nominal ENL while SLC starts from a single look.
    assert grd["looks_nominal"] == pytest.approx(rad.S1_IW_GRDH_NOMINAL_ENL)
    assert slc["looks_range"] > 1.0
    assert rad.nominal_looks(40.0, "SLC")["looks_nominal"] > slc["looks_nominal"]


def test_change_statistic_is_near_zero_under_the_null():
    rng = np.random.default_rng(5)
    a = rng.gamma(4.4, 1.0 / 4.4, 2000)
    b = rng.gamma(4.4, 1.0 / 4.4, 2000)
    result = rad.gamma_change_statistic(a, b, 4.4, 4.4)
    assert abs(result["log_ratio_db"]) < 0.3
    assert result["statistic"] < 10.0


def test_change_statistic_detects_a_real_change():
    rng = np.random.default_rng(6)
    a = rng.gamma(4.4, 1.0 / 4.4, 2000)
    b = rng.gamma(4.4, 2.0 / 4.4, 2000)  # 3 dB brighter
    result = rad.gamma_change_statistic(a, b, 4.4, 4.4)
    assert result["log_ratio_db"] == pytest.approx(3.0, abs=0.3)
    assert result["statistic"] > 100.0


def test_change_statistic_scales_with_sample_size():
    """
    The same decibel change is more significant with more effective looks.
    This is why a fixed decibel threshold calibrated on large fields is wrong
    on small ones, and why look counts are tracked per observation.
    """
    rng = np.random.default_rng(8)
    small_a = rng.gamma(4.4, 1.0 / 4.4, 30)
    small_b = rng.gamma(4.4, 1.3 / 4.4, 30)
    large_a = rng.gamma(4.4, 1.0 / 4.4, 3000)
    large_b = rng.gamma(4.4, 1.3 / 4.4, 3000)

    small = rad.gamma_change_statistic(small_a, small_b, 4.4, 4.4)
    large = rad.gamma_change_statistic(large_a, large_b, 4.4, 4.4)
    assert large["statistic"] > small["statistic"]
