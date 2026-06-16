"""
    Climate-science specific plots, operating along the time/batch axis.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import to_canonical
from ._style import apply_style, maybe_save

__all__ = ["plot_temporal_trend", "plot_temporal_scatter"]


def _spatial_mean_series(data: np.ndarray, ignore_nan: bool) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Computes, for a canonical (batch, H, W, C, T) array, the spatial
        mean and spatial standard deviation as a function of the merged
        (batch, time) sample axis, averaged over channels.

        Parameters:
        -----------
        - data : np.ndarray
            Canonical array of shape (batch, H, W, C, T).
        - ignore_nan : bool
            If True, uses nan-aware reductions.

        Returns:
        --------
        - (mean_series, std_series) : tuple of np.ndarray
            1D arrays of length (batch * T), in temporal/sample order.
    '''

    B, H, W, C, T = data.shape
    # (B, H, W, C, T) -> (B, T, H, W, C) -> (B*T, H, W, C)
    samples = np.moveaxis(data, 4, 1).reshape(B * T, H, W, C)

    mean_fn = np.nanmean if ignore_nan else np.mean
    std_fn = np.nanstd if ignore_nan else np.std

    mean_series = mean_fn(samples, axis=(1, 2, 3))
    std_series = std_fn(samples, axis=(1, 2, 3))

    return mean_series, std_series


def plot_temporal_trend(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    show_spread: bool = True,
    title: str = "Spatial-mean temporal trend",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Plots the spatial-mean time series of ground truth and prediction,
        optionally with a shaded +-1 std spatial-variability band. Useful for
        checking whether a model preserves the temporal/seasonal trend.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from spatial statistics (default: False).
        - show_spread : bool
            If True, shades +-1 spatial standard deviation around each mean line (default: True).
        - title : str
            Plot title.
        - save_path : str
            If given, the figure is saved to this path (default: None).
        - ax : matplotlib.axes.Axes
            Existing axis to draw on. If None, a new figure/axis is created.

        Returns:
        --------
        - (fig, ax) : tuple
            The matplotlib figure and axis, for further customization.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_temporal_trend

        gt   = np.random.rand(64, 64, 1, 365)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        fig, ax = plot_temporal_trend(gt, pred, data_format="hwct")
        ```
    '''

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    true_mean, true_std = _spatial_mean_series(y_true_c, ignore_nan)
    pred_mean, pred_std = _spatial_mean_series(y_pred_c, ignore_nan)

    x = np.arange(len(true_mean))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    if show_spread:
        ax.fill_between(x, true_mean - true_std, true_mean + true_std, color="black", alpha=0.12, label="GT +-1 sigma")
        ax.fill_between(x, pred_mean - pred_std, pred_mean + pred_std, color="#D65F5F", alpha=0.12, label="Pred +-1 sigma")

    ax.plot(x, true_mean, color="black", lw=1.8, label="Ground truth mean")
    ax.plot(x, pred_mean, color="#D65F5F", lw=1.8, ls="--", label="Prediction mean")

    ax.set_xlabel("Sample index (temporal order)")
    ax.set_ylabel("Spatial mean")
    ax.set_title(title)
    ax.legend(fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax


def plot_temporal_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Spatial-mean scatter",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Scatter plot of the spatial-mean ground truth vs spatial-mean
        prediction, one point per sample (batch/time index), with a 1:1
        reference line.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from spatial statistics (default: False).
        - title : str
            Plot title.
        - save_path : str
            If given, the figure is saved to this path (default: None).
        - ax : matplotlib.axes.Axes
            Existing axis to draw on. If None, a new figure/axis is created.

        Returns:
        --------
        - (fig, ax) : tuple
            The matplotlib figure and axis, for further customization.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_temporal_scatter

        gt   = np.random.rand(64, 64, 1, 365)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        fig, ax = plot_temporal_scatter(gt, pred, data_format="hwct")
        ```
    '''

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    true_mean, _ = _spatial_mean_series(y_true_c, ignore_nan)
    pred_mean, _ = _spatial_mean_series(y_pred_c, ignore_nan)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(true_mean, pred_mean, s=10, alpha=0.6, color="#4878CF")
    lims = [float(min(true_mean.min(), pred_mean.min())), float(max(true_mean.max(), pred_mean.max()))]
    ax.plot(lims, lims, "r--", lw=1.2, label="1:1")
    ax.set_xlabel("Ground truth spatial mean")
    ax.set_ylabel("Prediction spatial mean")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    maybe_save(fig, save_path)

    return fig, ax
