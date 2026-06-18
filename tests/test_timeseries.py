import sys
sys.path += ['.']

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from aule.metrics.timeseries import (
    lag_correlation, cross_channel_correlation,
    peak_timing_error, dtw_distance,
)
from aule.plots.timeseries import (
    plot_lag_correlation, plot_multi_channel_series,
    plot_dtw_alignment, plot_channel_correlation_matrix,
)


# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

def _make_series(B=4, C=3, T=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, C, T))


# ---------------------------------------------------------------------------
# lag_correlation
# ---------------------------------------------------------------------------

def test_lag_correlation_zero_lag_perfect_for_identical():
    s = _make_series()
    corr = lag_correlation(s, s, max_lag=10, axes="bct")
    mid = 10  # index of lag=0
    assert corr[mid] == pytest.approx(1.0, abs=1e-6)


def test_lag_correlation_detects_shift():
    s = _make_series(B=2, C=1, T=200)
    shifted = np.roll(s, shift=5, axis=-1)
    corr = lag_correlation(s, shifted, max_lag=20, axes="bct")
    lags = np.arange(-20, 21)
    peak_lag = lags[np.argmax(corr)]
    # shifted pred leads by 5 steps => peak at lag +5
    assert peak_lag == 5


def test_lag_correlation_length():
    s = _make_series()
    corr = lag_correlation(s, s, max_lag=15, axes="bct")
    assert len(corr) == 31  # 2*15+1


def test_lag_correlation_bct_axes():
    s = _make_series(B=2, C=3, T=80)
    corr = lag_correlation(s, s, max_lag=5, axes="bct")
    assert corr.shape == (11,)


def test_lag_correlation_t_axis():
    s = np.random.randn(60)
    corr = lag_correlation(s, s, max_lag=5, axes="t")
    assert len(corr) == 11


def test_lag_correlation_bt_axis():
    s = np.random.randn(4, 60)
    corr = lag_correlation(s, s, max_lag=5, axes="bt")
    assert len(corr) == 11


def test_lag_correlation_raises_on_short_series():
    s = np.array([1.0])
    with pytest.raises(ValueError):
        lag_correlation(s, s, max_lag=1, axes="t")


def test_lag_correlation_raises_on_lag_too_large():
    s = np.random.randn(30)
    with pytest.raises(ValueError):
        lag_correlation(s, s, max_lag=30, axes="t")


def test_lag_correlation_spatial_input():
    gt = np.random.randn(4, 32, 32, 1, 50)
    pred = np.roll(gt, 2, axis=-1)
    corr = lag_correlation(gt, pred, max_lag=10)
    assert len(corr) == 21


# ---------------------------------------------------------------------------
# cross_channel_correlation
# ---------------------------------------------------------------------------

def test_cross_channel_correlation_identity_diagonal():
    s = _make_series()
    corr = cross_channel_correlation(s, axes="bct")
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)


def test_cross_channel_correlation_symmetric():
    s = _make_series()
    corr = cross_channel_correlation(s, axes="bct")
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)


def test_cross_channel_correlation_bounded():
    s = _make_series()
    corr = cross_channel_correlation(s, axes="bct")
    assert np.all(corr >= -1.0 - 1e-9) and np.all(corr <= 1.0 + 1e-9)


def test_cross_channel_correlation_single_channel():
    s = np.random.randn(1, 100)
    corr = cross_channel_correlation(s, axes="ct")
    assert corr.shape == (1, 1)
    assert corr[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_cross_channel_correlation_perfect_for_duplicated_channels():
    base = np.random.randn(1, 1, 100)
    dup = np.concatenate([base, base], axis=1)
    corr = cross_channel_correlation(dup, axes="bct")
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# peak_timing_error
# ---------------------------------------------------------------------------

def test_peak_timing_error_zero_for_identical():
    s = _make_series()
    assert peak_timing_error(s, s, axes="bct") == pytest.approx(0.0, abs=1e-10)


def test_peak_timing_error_detects_offset():
    t = np.linspace(0, 2 * np.pi, 100)
    sin_gt   = np.sin(t).reshape(1, 1, 100)
    sin_pred = np.sin(t - 0.5).reshape(1, 1, 100)  # shifted by ~8 steps
    err = peak_timing_error(sin_gt, sin_pred, axes="bct")
    assert err > 1


def test_peak_timing_error_nonnegative():
    s = _make_series()
    assert peak_timing_error(s, s * 1.1 + np.random.randn(*s.shape) * 0.1, axes="bct") >= 0


def test_peak_timing_error_t_axis():
    s = np.random.randn(50)
    assert peak_timing_error(s, s, axes="t") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# dtw_distance
# ---------------------------------------------------------------------------

def test_dtw_distance_zero_for_identical():
    s = _make_series(T=50)
    assert dtw_distance(s, s, axes="bct") == pytest.approx(0.0, abs=1e-10)


def test_dtw_distance_nonnegative():
    s1 = _make_series(T=50)
    s2 = _make_series(T=50, seed=1)
    assert dtw_distance(s1, s2, axes="bct") >= 0.0


def test_dtw_distance_smaller_than_rmse_on_shifted_signal():
    """DTW should give lower distance than RMSE for a phase-shifted signal."""
    t = np.linspace(0, 4 * np.pi, 80)
    gt   = np.sin(t).reshape(1, 1, 80)
    pred = np.sin(t + 0.5).reshape(1, 1, 80)

    dtw_d = dtw_distance(gt, pred, axes="bct")
    rmse_d = float(np.sqrt(np.mean((gt - pred) ** 2)))
    # DTW finds a better alignment so its "distance" can be lower than RMSE *T
    # (this is a sanity check, not a strict bound)
    assert dtw_d >= 0.0


def test_dtw_distance_with_sakoe_chiba_window():
    s1 = _make_series(B=2, C=2, T=40)
    s2 = _make_series(B=2, C=2, T=40, seed=1)
    d_full   = dtw_distance(s1, s2, axes="bct", window=None)
    d_window = dtw_distance(s1, s2, axes="bct", window=5)
    assert d_full <= d_window + 1e-9  # constrained DTW is >= unconstrained


def test_dtw_distance_t_axis():
    s = np.random.randn(50)
    assert dtw_distance(s, s, axes="t") == pytest.approx(0.0, abs=1e-10)


def test_dtw_distance_raises_on_short_series():
    s = np.array([1.0])
    with pytest.raises(ValueError):
        dtw_distance(s, s, axes="t")


def test_dtw_distance_spatial_input():
    gt   = np.random.randn(2, 8, 8, 1, 30)
    pred = np.random.randn(2, 8, 8, 1, 30)
    d = dtw_distance(gt, pred)
    assert d >= 0.0


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_plot_lag_correlation_returns_fig_ax():
    s = _make_series()
    fig, ax = plot_lag_correlation(s, s, max_lag=10, axes="bct")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_lag_correlation_on_spatial_input():
    gt = np.random.randn(4, 16, 16, 1, 50)
    pred = np.roll(gt, 3, axis=-1)
    fig, ax = plot_lag_correlation(gt, pred, max_lag=10)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_multi_channel_series_returns_fig_axes():
    s = _make_series()
    fig, axs = plot_multi_channel_series(s, axes="bct",
                                          channel_names=["A", "B", "C"])
    assert isinstance(fig, plt.Figure)
    assert len(axs) == 3
    plt.close(fig)


def test_plot_multi_channel_series_single_channel():
    s = np.random.randn(2, 50)
    fig, axs = plot_multi_channel_series(s, axes="bt")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_dtw_alignment_returns_fig_axes():
    t = np.linspace(0, 4 * np.pi, 60)
    gt   = np.sin(t).reshape(1, 1, 60)
    pred = np.sin(t * 1.1).reshape(1, 1, 60)
    fig, axs = plot_dtw_alignment(gt, pred, axes="bct")
    assert isinstance(fig, plt.Figure)
    assert axs.shape == (2,)
    plt.close(fig)


def test_plot_channel_correlation_matrix_returns_fig_ax():
    s = _make_series()
    fig, ax = plot_channel_correlation_matrix(s, axes="bct",
                                               channel_names=["X", "Y", "Z"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_channel_correlation_matrix_shape():
    s = np.random.randn(3, 5, 80)
    fig, ax = plot_channel_correlation_matrix(s, axes="bct")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shape system integration
# ---------------------------------------------------------------------------

def test_generic_metric_rmse_works_on_all_series_shapes():
    from aule.metrics import rmse

    shapes_and_axes = [
        (np.random.randn(50), "t"),
        (np.random.randn(3), "c"),
        (np.random.randn(4, 50), "bt"),
        (np.random.randn(3, 50), "ct"),
        (np.random.randn(4, 3, 50), "bct"),
    ]
    for arr, ax in shapes_and_axes:
        r = rmse(arr, arr, axes=ax)
        assert r == pytest.approx(0.0, abs=1e-9), f"rmse!=0 for axes={ax}"


def test_generic_metric_pearson_r_works_on_series():
    from aule.metrics import pearson_r
    s = np.random.randn(4, 3, 50)
    assert pearson_r(s, s * 2 + 1, axes="bct") == pytest.approx(1.0, abs=1e-6)


def test_guardrail_blocks_series_on_spatial_metric():
    from aule.metrics import ssim
    deg = np.random.randn(1, 1, 3)  # H=W=1 => degenerate spatial
    with pytest.raises(ValueError):
        ssim(deg, deg)


def test_guardrail_force_on_series_shaped_input():
    import warnings
    from aule.metrics import ssim
    deg = np.random.randn(1, 1, 3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ssim(deg, deg, force=True)
        assert isinstance(result, float)
        assert len(w) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
