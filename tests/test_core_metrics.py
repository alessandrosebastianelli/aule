import sys
sys.path += ['.']

import numpy as np
import pytest

from aule.metrics import (
    rmse, mae, bias, pearson_r, ssim, mse, psnr, r2_score, mape, smape, nse, kge,
    max_error, explained_variance, wasserstein_distance, quantile_mapping_bias,
)


def test_rmse_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert rmse(x, x) == pytest.approx(0.0, abs=1e-10)


def test_mae_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert mae(x, x) == pytest.approx(0.0, abs=1e-10)


def test_bias_sign():
    gt = np.zeros((16, 16, 1))
    pred = np.full((16, 16, 1), 0.5)
    assert bias(gt, pred) == pytest.approx(0.5, abs=1e-10)


def test_pearson_r_perfect_correlation():
    gt = np.random.rand(16, 16, 1)
    pred = gt * 2.0 + 1.0
    assert pearson_r(gt, pred) == pytest.approx(1.0, abs=1e-6)


def test_pearson_r_with_nan_ignored():
    gt = np.array([[[1.0]], [[2.0]], [[3.0]], [[np.nan]]])
    pred = np.array([[[1.0]], [[2.0]], [[3.0]], [[5.0]]])
    r = pearson_r(gt, pred, ignore_nan=True)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_ssim_perfect_for_identical_field():
    x = np.random.rand(32, 32, 1)
    score = ssim(x, x)
    assert score == pytest.approx(1.0, abs=1e-3)


def test_rmse_with_4d_hwct_format():
    gt = np.random.rand(32, 32, 1, 5)
    pred = gt + 0.0
    assert rmse(gt, pred, data_format="hwct") == pytest.approx(0.0, abs=1e-10)


def test_mse_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert mse(x, x) == pytest.approx(0.0, abs=1e-10)


def test_mse_equals_rmse_squared():
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    assert mse(gt, pred) == pytest.approx(rmse(gt, pred) ** 2, rel=1e-9)


def test_psnr_infinite_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert psnr(x, x) == float("inf")


def test_psnr_decreases_with_more_noise():
    gt = np.random.rand(8, 32, 32, 1)
    pred_low_noise = gt + np.random.normal(0, 0.01, gt.shape)
    pred_high_noise = gt + np.random.normal(0, 0.2, gt.shape)
    assert psnr(gt, pred_low_noise, data_range=1.0) > psnr(gt, pred_high_noise, data_range=1.0)


def test_r2_score_perfect_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert r2_score(x, x) == pytest.approx(1.0, abs=1e-10)


def test_mape_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1) + 1.0
    assert mape(x, x) == pytest.approx(0.0, abs=1e-8)


def test_smape_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert smape(x, x) == pytest.approx(0.0, abs=1e-8)


def test_smape_bounded():
    gt = np.random.rand(8, 32, 32, 1)
    pred = np.random.rand(8, 32, 32, 1) * 10
    score = smape(gt, pred)
    assert 0.0 <= score <= 200.0


def test_nse_equals_r2_score():
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    assert nse(gt, pred) == pytest.approx(r2_score(gt, pred), abs=1e-10)


def test_kge_perfect_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    result = kge(x, x)
    assert result["kge"] == pytest.approx(1.0, abs=1e-6)
    assert result["r"] == pytest.approx(1.0, abs=1e-6)
    assert result["alpha"] == pytest.approx(1.0, abs=1e-6)
    assert result["beta"] == pytest.approx(1.0, abs=1e-6)


def test_max_error_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert max_error(x, x) == pytest.approx(0.0, abs=1e-10)


def test_max_error_at_least_mae():
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    assert max_error(gt, pred) >= mae(gt, pred)


def test_explained_variance_perfect_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert explained_variance(x, x) == pytest.approx(1.0, abs=1e-10)


def test_wasserstein_distance_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert wasserstein_distance(x, x) == pytest.approx(0.0, abs=1e-10)


def test_wasserstein_distance_nonnegative():
    gt = np.random.exponential(1.0, (8, 32, 32, 1))
    pred = np.random.exponential(1.5, (8, 32, 32, 1))
    assert wasserstein_distance(gt, pred) >= 0.0


def test_quantile_mapping_bias_zero_for_identical_arrays():
    x = np.random.rand(8, 32, 32, 1)
    assert quantile_mapping_bias(x, x) == pytest.approx(0.0, abs=1e-10)


def test_quantile_mapping_bias_detects_scale_shift():
    gt = np.random.exponential(1.0, (8, 32, 32, 1))
    pred = gt * 1.5
    score = quantile_mapping_bias(gt, pred)
    assert score > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
