import sys
sys.path += ['.']

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from aule.plots._style import (
    power_norm, symlog_norm, asymmetric_twoslope_norm, resolve_diverging_norm,
)
from aule.plots import plot_field_comparison, plot_bias_map, plot_error_map


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path


# --- power_norm ---

def test_power_norm_midpoint_maps_to_half():
    data = np.array([-1.0, 0.0, 1.0])
    norm = power_norm(data, gamma=0.5)
    assert norm(0.0) == pytest.approx(0.5, abs=1e-6)


def test_power_norm_symmetric_around_center():
    norm = power_norm(vmin=-1.0, vmax=1.0, gamma=0.4)
    above = norm(0.5) - 0.5
    below = 0.5 - norm(-0.5)
    assert above == pytest.approx(below, abs=1e-6)


def test_power_norm_default_range_from_data():
    data = np.array([-0.4, 0.0, 0.8, np.nan])
    norm = power_norm(data, gamma=0.5)
    assert norm.vmin == pytest.approx(-0.8, abs=1e-6)
    assert norm.vmax == pytest.approx(0.8, abs=1e-6)


def test_power_norm_compresses_near_zero_more_than_linear():
    # a value at 25% of the range should map to MORE than 25% of the
    # colorbar under gamma<1 power compression (extremes get more contrast)
    norm = power_norm(vmin=-1.0, vmax=1.0, gamma=0.5)
    linear_position = 0.625  # (0.25 - (-1)) / (1 - (-1))
    assert norm(0.25) > linear_position


# --- symlog_norm ---

def test_symlog_norm_zero_maps_to_half():
    norm = symlog_norm(vmin=-1.0, vmax=1.0, linthresh=0.05)
    assert norm(0.0) == pytest.approx(0.5, abs=1e-6)


def test_symlog_norm_auto_linthresh_from_data():
    data = np.random.normal(0, 0.05, (200, 200))
    norm = symlog_norm(data)
    assert norm.linthresh > 0


def test_symlog_norm_explicit_linthresh_respected():
    norm = symlog_norm(vmin=-1.0, vmax=1.0, linthresh=0.07)
    assert norm.linthresh == pytest.approx(0.07, abs=1e-9)


# --- asymmetric_twoslope_norm ---

def test_asymmetric_twoslope_norm_vcenter_maps_to_half():
    data = np.array([-0.3, 0.0, 1.5])
    norm = asymmetric_twoslope_norm(data, vcenter=0.0)
    assert norm(0.0) == pytest.approx(0.5, abs=1e-6)


def test_asymmetric_twoslope_norm_uses_data_min_max():
    data = np.array([-0.3, 0.0, 1.5, np.nan])
    norm = asymmetric_twoslope_norm(data, vcenter=0.0)
    assert norm.vmin == pytest.approx(-0.3, abs=1e-6)
    assert norm.vmax == pytest.approx(1.5, abs=1e-6)


def test_asymmetric_twoslope_norm_nonzero_vcenter():
    norm = asymmetric_twoslope_norm(vmin=0.0, vmax=10.0, vcenter=2.0)
    assert norm(2.0) == pytest.approx(0.5, abs=1e-6)


# --- resolve_diverging_norm dispatcher ---

def test_resolve_diverging_norm_linear():
    data = np.random.normal(0, 0.1, (50, 50))
    norm = resolve_diverging_norm(data, norm_type="linear")
    assert hasattr(norm, "vcenter")


def test_resolve_diverging_norm_power():
    data = np.random.normal(0, 0.1, (50, 50))
    norm = resolve_diverging_norm(data, norm_type="power", gamma=0.4)
    assert norm(0.0) == pytest.approx(0.5, abs=1e-6)


def test_resolve_diverging_norm_symlog():
    data = np.random.normal(0, 0.1, (50, 50))
    norm = resolve_diverging_norm(data, norm_type="symlog", linthresh=0.02)
    assert norm.linthresh == pytest.approx(0.02, abs=1e-9)


def test_resolve_diverging_norm_twoslope():
    data = np.random.normal(0, 0.1, (50, 50))
    norm = resolve_diverging_norm(data, norm_type="twoslope", vcenter=0.0)
    assert norm(0.0) == pytest.approx(0.5, abs=1e-6)


def test_resolve_diverging_norm_invalid_raises():
    data = np.random.normal(0, 0.1, (50, 50))
    with pytest.raises(ValueError):
        resolve_diverging_norm(data, norm_type="not_a_real_type")


# --- integration with plotting functions ---

def test_plot_field_comparison_with_power_diff_norm(out_dir):
    gt = np.random.rand(32, 32, 1)
    pred = gt + np.random.normal(0, 0.05, gt.shape)
    fig, axes = plot_field_comparison(
        gt, pred, diff_norm_type="power", diff_norm_kwargs={"gamma": 0.4},
        save_path=os.path.join(out_dir, "field_power.png"),
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_bias_map_with_symlog_norm(out_dir):
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    fig, ax = plot_bias_map(
        gt, pred, norm_type="symlog", norm_kwargs={"linthresh": 0.02},
        save_path=os.path.join(out_dir, "bias_symlog.png"),
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_error_map_signed_with_twoslope_norm(out_dir):
    gt = np.random.rand(32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    fig, ax = plot_error_map(
        gt, pred, abs_error=False, norm_type="twoslope", norm_kwargs={"vcenter": 0.0},
        save_path=os.path.join(out_dir, "error_twoslope.png"),
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_bias_map_invalid_norm_type_raises():
    gt = np.random.rand(8, 32, 32, 1)
    pred = gt + np.random.normal(0, 0.1, gt.shape)
    with pytest.raises(ValueError):
        plot_bias_map(gt, pred, norm_type="invalid")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
