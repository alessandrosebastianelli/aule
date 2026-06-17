import sys
sys.path += ['.']

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest

from aule.report import generate_report


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path


def test_generate_report_creates_file(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    path = os.path.join(out_dir, "report.html")
    result = generate_report(gt, pred, save_path=path)
    assert result == path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_generate_report_contains_metrics_and_plots(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    path = os.path.join(out_dir, "report.html")
    generate_report(gt, pred, save_path=path)

    with open(path) as f:
        content = f.read()

    assert "RMSE" in content
    assert "<img" in content
    assert content.count("<img") >= 7


def test_generate_report_3d_input_no_time_axis(out_dir):
    gt = np.random.rand(32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    path = os.path.join(out_dir, "report3d.html")
    generate_report(gt, pred, save_path=path)
    assert os.path.exists(path)


def test_generate_report_with_time_axis_includes_climate_metrics(out_dir):
    gt = np.random.rand(16, 16, 1, 20)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    path = os.path.join(out_dir, "report_time.html")
    generate_report(gt, pred, save_path=path, data_format="hwct")

    with open(path) as f:
        content = f.read()

    assert "Trend error" in content
    assert "Seasonal error" in content


def test_generate_report_handles_nan_with_ignore_nan(out_dir):
    gt = np.random.rand(32, 32, 1)
    gt[:5, :5] = np.nan
    pred = np.random.rand(32, 32, 1)
    path = os.path.join(out_dir, "report_nan.html")
    # should not raise
    generate_report(gt, pred, save_path=path, ignore_nan=True)
    assert os.path.exists(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
