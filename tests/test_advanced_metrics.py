import sys
sys.path += ['.']

import numpy as np
import pytest

from aule.metrics import (
    spectral_error, gradient_error,
    seasonal_error, percentile_error, pixelwise_temporal_correlation,
    ensemble_spread, crps, rank_histogram,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
