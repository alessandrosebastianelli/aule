import sys
sys.path += ['.']

import numpy as np
import pytest

from aule.metrics import (
    spectral_error, gradient_error, psd_radial_error, spectral_angle_mapper,
    seasonal_error, percentile_error, pixelwise_temporal_correlation,
    trend_error, extreme_event_duration_error, autocorrelation_error,
    ensemble_spread, crps, rank_histogram, brier_score, spread_skill_ratio, crps_skill_score,
    normalized_difference_index, index_error, change_detection_error,
)


def test_spectral_error_zero_for_identical():
    x = np.random.rand(16, 32, 32, 1)
    assert spectral_error(x, x) == pytest.approx(0.0, abs=1e-10)


def test_gradient_error_zero_for_identical():
    x = np.random.rand(32, 32, 1)
    assert gradient_error(x, x) == pytest.approx(0.0, abs=1e-10)


def test_seasonal_error_zero_for_identical():
    x = np.random.rand(32, 32, 1, 10)
    assert seasonal_error(x, x, data_format="hwct") == pytest.approx(0.0, abs=1e-10)


def test_percentile_error_zero_for_identical():
    x = np.random.rand(8, 32, 32, 1)
    assert percentile_error(x, x, percentile=95.0) == pytest.approx(0.0, abs=1e-10)


def test_pixelwise_temporal_correlation_perfect():
    gt = np.random.rand(50, 16, 16, 1)
    pred = gt * 2.0 + 1.0
    r_map = pixelwise_temporal_correlation(gt, pred)
    assert r_map.shape == (16, 16, 1)
    assert np.all(r_map > 0.99)


def test_pixelwise_temporal_correlation_requires_multiple_samples():
    gt = np.random.rand(16, 16, 1)
    pred = np.random.rand(16, 16, 1)
    with pytest.raises(ValueError):
        pixelwise_temporal_correlation(gt, pred)


def test_ensemble_spread_zero_for_identical_members():
    member = np.random.rand(16, 16, 1)
    ensemble = np.stack([member] * 5, axis=0)
    assert ensemble_spread(ensemble) == pytest.approx(0.0, abs=1e-10)


def test_crps_zero_for_perfect_deterministic_ensemble():
    gt = np.random.rand(16, 16, 1)
    ensemble = np.stack([gt] * 5, axis=0)
    assert crps(gt, ensemble) == pytest.approx(0.0, abs=1e-6)


def test_rank_histogram_length():
    gt = np.random.rand(16, 16, 1)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    counts = rank_histogram(gt, ensemble)
    assert len(counts) == 11
    assert counts.sum() == gt.size


def test_normalized_difference_index_range():
    a = np.random.rand(16, 16, 1)
    b = np.random.rand(16, 16, 1)
    index = normalized_difference_index(a, b)
    assert np.all(index >= -1.0 - 1e-6) and np.all(index <= 1.0 + 1e-6)


def test_index_error_zero_for_identical_bands():
    a = np.random.rand(16, 16, 1) + 0.1
    b = np.random.rand(16, 16, 1) + 0.1
    assert index_error(a, b, a, b) == pytest.approx(0.0, abs=1e-10)


def test_change_detection_error_zero_for_identical():
    x = np.random.rand(16, 16, 1, 6)
    assert change_detection_error(x, x, data_format="hwct") == pytest.approx(0.0, abs=1e-10)


def test_change_detection_error_requires_time_axis():
    x = np.random.rand(16, 16, 1)
    with pytest.raises(ValueError):
        change_detection_error(x, x)


def test_psd_radial_error_zero_for_identical():
    x = np.random.rand(8, 32, 32, 1)
    assert psd_radial_error(x, x) == pytest.approx(0.0, abs=1e-10)


def test_spectral_angle_mapper_zero_for_identical():
    x = np.random.rand(32, 32, 4) + 0.1
    assert spectral_angle_mapper(x, x) == pytest.approx(0.0, abs=1e-3)


def test_spectral_angle_mapper_in_valid_range():
    a = np.random.rand(32, 32, 4) + 0.1
    b = np.random.rand(32, 32, 4) + 0.1
    angle = spectral_angle_mapper(a, b)
    assert 0.0 <= angle <= 180.0


def test_trend_error_zero_for_identical():
    t = np.arange(50)
    x = np.tile((0.02 * t).reshape(1, 1, 1, -1), (16, 16, 1, 1))
    assert trend_error(x, x, data_format="hwct") == pytest.approx(0.0, abs=1e-6)


def test_trend_error_requires_time_axis():
    x = np.random.rand(16, 16, 1)
    with pytest.raises(ValueError):
        trend_error(x, x)


def test_extreme_event_duration_error_zero_for_identical():
    x = np.random.rand(16, 16, 1, 40) + 0.5
    assert extreme_event_duration_error(x, x, threshold=0.8, data_format="hwct") == pytest.approx(0.0, abs=1e-10)


def test_extreme_event_duration_error_no_events():
    x = np.zeros((16, 16, 1, 20))
    assert extreme_event_duration_error(x, x, threshold=5.0, data_format="hwct") == pytest.approx(0.0, abs=1e-10)


def test_autocorrelation_error_zero_for_identical():
    x = np.cumsum(np.random.randn(16, 16, 1, 100), axis=-1)
    assert autocorrelation_error(x, x, max_lag=5, data_format="hwct") == pytest.approx(0.0, abs=1e-10)


def test_autocorrelation_error_requires_enough_steps():
    x = np.random.rand(16, 16, 1, 5)
    with pytest.raises(ValueError):
        autocorrelation_error(x, x, max_lag=10, data_format="hwct")


def test_brier_score_zero_for_perfect_deterministic_forecast():
    gt = np.random.rand(16, 16, 1) + 0.5
    ensemble = np.stack([gt] * 5, axis=0)
    assert brier_score(gt, ensemble, threshold=0.5) == pytest.approx(0.0, abs=1e-10)


def test_brier_score_bounded():
    gt = np.random.rand(16, 16, 1)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    score = brier_score(gt, ensemble, threshold=0.5)
    assert 0.0 <= score <= 1.0


def test_spread_skill_ratio_positive():
    gt = np.random.rand(16, 16, 1)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    ratio = spread_skill_ratio(gt, ensemble)
    assert ratio >= 0.0


def test_crps_skill_score_positive_when_forecast_better():
    gt = np.random.rand(16, 16, 1)
    good_forecast = gt[np.newaxis] + np.random.normal(0, 0.05, (10, 16, 16, 1))
    bad_reference = gt[np.newaxis] + np.random.normal(0, 0.3, (10, 16, 16, 1))
    score = crps_skill_score(gt, good_forecast, bad_reference)
    assert score > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
