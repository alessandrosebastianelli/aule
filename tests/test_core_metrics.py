import sys
sys.path += ['.']

import numpy as np
import pytest

from aule.metrics import rmse, mae, bias, pearson_r, ssim


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
