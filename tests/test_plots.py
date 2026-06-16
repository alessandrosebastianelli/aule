import sys
sys.path += ['.']

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from aule.plots import (
    plot_scatter, plot_qq, plot_histogram_comparison, plot_error_histogram,
    plot_field_comparison, plot_bias_map, plot_correlation_map,
    plot_temporal_trend, plot_temporal_scatter,
    plot_ensemble_spread_map, plot_rank_histogram,
)
from aule.metrics import pixelwise_temporal_correlation, rank_histogram


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path


def test_plot_scatter_returns_fig_ax_and_saves(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    path = os.path.join(out_dir, "scatter.png")
    fig, ax = plot_scatter(gt, pred, save_path=path)
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(path)
    plt.close(fig)


def test_plot_qq(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, ax = plot_qq(gt, pred, save_path=os.path.join(out_dir, "qq.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_histogram_comparison(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, ax = plot_histogram_comparison(gt, pred, save_path=os.path.join(out_dir, "hist.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_error_histogram(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, ax = plot_error_histogram(gt, pred, save_path=os.path.join(out_dir, "err_hist.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_field_comparison_no_geo(out_dir):
    gt = np.random.rand(32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, axes = plot_field_comparison(gt, pred, save_path=os.path.join(out_dir, "field.png"))
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 3
    plt.close(fig)


def test_plot_bias_map_no_geo(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, ax = plot_bias_map(gt, pred, save_path=os.path.join(out_dir, "bias.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_map_no_geo(out_dir):
    gt = np.random.rand(50, 16, 16, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    r_map = pixelwise_temporal_correlation(gt, pred)
    fig, ax = plot_correlation_map(r_map, save_path=os.path.join(out_dir, "corr.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_temporal_trend(out_dir):
    gt = np.random.rand(32, 32, 1, 20)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, ax = plot_temporal_trend(gt, pred, data_format="hwct", save_path=os.path.join(out_dir, "trend.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_temporal_scatter(out_dir):
    gt = np.random.rand(32, 32, 1, 20)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, ax = plot_temporal_scatter(gt, pred, data_format="hwct", save_path=os.path.join(out_dir, "temp_scatter.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ensemble_spread_map_no_geo(out_dir):
    ensemble = np.random.rand(10, 32, 32, 1)
    fig, ax = plot_ensemble_spread_map(ensemble, save_path=os.path.join(out_dir, "spread.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rank_histogram(out_dir):
    gt = np.random.rand(16, 16, 1)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    counts = rank_histogram(gt, ensemble)
    fig, ax = plot_rank_histogram(counts, save_path=os.path.join(out_dir, "rank.png"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_returns_axis_for_external_customization():
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, ax = plot_scatter(gt, pred)
    # external customization should work without errors
    ax.set_title("Custom title")
    ax.set_xlabel("Custom xlabel")
    plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
