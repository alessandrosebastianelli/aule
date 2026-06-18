"""
    Plots for pure time series data.

    These complement the spatial plots in `aule.plots.spatial` and the
    generic distribution plots in `aule.plots.core` for inputs in the
    series-family shapes: (T,), (C,), (B,T), (C,T), (B,C,T).
"""

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import to_canonical
from ._style import apply_style, maybe_save
from .._logging import logger

__all__ = [
    "plot_lag_correlation",
    "plot_multi_channel_series",
    "plot_dtw_alignment",
    "plot_channel_correlation_matrix",
]


def _spatial_mean(arr: np.ndarray, ignore_nan: bool) -> np.ndarray:
    if ignore_nan:
        return np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=(1, 2))
    return np.mean(arr, axis=(1, 2))


def plot_lag_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag: int = 20,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Cross-correlation by lag",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Plots the cross-correlation between ground truth and prediction
        as a bar chart over lags -max_lag … +max_lag.  The vertical dashed
        line marks lag 0; a peak to the right means the prediction leads
        the truth (the model is early), a peak to the left means it lags.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth, any spatial or series shape.
        - y_pred : np.ndarray
            Prediction, same shape as y_true.
        - max_lag : int
            Maximum absolute lag to plot (default: 20).
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - ignore_nan : bool
            If True, NaN values are excluded (default: False).
        - title : str
            Plot title.
        - save_path : str
            If given, saves the figure to this path.
        - ax : matplotlib.axes.Axes
            Existing axis to draw on.

        Returns:
        --------
        - (fig, ax) : tuple

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_lag_correlation

        gt   = np.random.randn(4, 2, 200)
        pred = np.roll(gt, 3, axis=-1) + 0.1 * np.random.randn(*gt.shape)
        fig, ax = plot_lag_correlation(gt, pred, max_lag=30, axes="bct")
        ```
    '''
    from ..metrics.timeseries import lag_correlation

    apply_style()

    corr = lag_correlation(y_true, y_pred, max_lag=max_lag,
                           data_format=data_format, axes=axes, ignore_nan=ignore_nan)
    lags = np.arange(-max_lag, max_lag + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    colors = ["#D65F5F" if c < 0 else "#4878CF" for c in corr]
    ax.bar(lags, corr, color=colors, alpha=0.85, edgecolor="none")
    ax.axvline(0, color="black", lw=1.0, ls="--", label="lag = 0")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Lag (time steps)  →  positive = prediction leads")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(title)
    ax.legend(fontsize=9)

    maybe_save(fig, save_path)
    return fig, ax


def plot_multi_channel_series(
    data: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    ignore_nan: bool = False,
    max_batch: int = 3,
    title: str = "Multi-channel time series",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    '''
        Plots each channel as a separate panel (subplot row), overlaying
        up to `max_batch` batch members as semi-transparent lines.  Useful
        for a quick visual inspection of multi-variable or multi-station
        time series.

        Parameters:
        -----------
        - data : np.ndarray
            Input array, any spatial or series shape.
        - channel_names : sequence of str
            Optional labels for each channel/panel.
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - ignore_nan : bool
            If True, NaN values are masked in each line (default: False).
        - max_batch : int
            Maximum number of batch members to overlay per panel (default: 3).
        - title : str
            Figure-level title.
        - save_path : str
            If given, saves the figure to this path.

        Returns:
        --------
        - (fig, axes_array) : tuple
            Figure and 1D array of axes, one per channel.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_multi_channel_series

        data = np.cumsum(np.random.randn(4, 3, 100), axis=-1)
        fig, axs = plot_multi_channel_series(data, axes="bct",
                                              channel_names=["Temp", "Precip", "Wind"])
        ```
    '''
    apply_style()
    logger.debug("plot_multi_channel_series: shape=%s, axes=%s", data.shape, axes)

    data_c = to_canonical(data, data_format=data_format, axes=axes)
    # (B, H, W, C, T) -> (B, C, T) via spatial mean
    bct = _spatial_mean(data_c.astype(np.float64), ignore_nan)
    B, C, T = bct.shape

    n_batch = min(B, max_batch)
    t = np.arange(T)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_batch))

    fig, axs = plt.subplots(C, 1, figsize=(10, 2.5 * C), sharex=True)
    if C == 1:
        axs = np.array([axs])

    for ci in range(C):
        for bi in range(n_batch):
            y = bct[bi, ci]
            if ignore_nan:
                y = np.where(np.isfinite(y), y, np.nan)
            label = f"batch {bi}" if n_batch > 1 else None
            axs[ci].plot(t, y, color=colors[bi], alpha=0.75, lw=1.4, label=label)

        name = channel_names[ci] if channel_names and ci < len(channel_names) else f"channel {ci}"
        axs[ci].set_ylabel(name)
        axs[ci].grid(True, alpha=0.2)
        if n_batch > 1 and ci == 0:
            axs[ci].legend(fontsize=8, loc="upper right")

    axs[-1].set_xlabel("Time step")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    maybe_save(fig, save_path)
    return fig, axs


def plot_dtw_alignment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batch_index: int = 0,
    channel_index: int = 0,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    window: Optional[int] = None,
    title: str = "DTW alignment",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    '''
        Plots the DTW alignment between ground truth and prediction for one
        (batch, channel) pair.  Left panel: the cost matrix with the
        optimal warping path overlaid.  Right panel: both series with
        alignment lines connecting matched time steps.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth, any spatial or series shape.
        - y_pred : np.ndarray
            Prediction, same shape as y_true.
        - batch_index, channel_index : int
            Which (batch, channel) slice to visualise.
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - window : int
            Sakoe-Chiba band width.  None = unconstrained (default: None).
        - title : str
            Figure-level title.
        - save_path : str
            If given, saves the figure to this path.

        Returns:
        --------
        - (fig, axes_array) : tuple
            Figure and array of 2 axes [cost matrix, series alignment].

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_dtw_alignment

        t = np.linspace(0, 4*np.pi, 80)
        gt   = np.sin(t).reshape(1, 1, 80)
        pred = np.sin(t * 1.15).reshape(1, 1, 80)
        fig, axs = plot_dtw_alignment(gt, pred, axes="bct")
        ```
    '''
    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format, axes=axes)
    y_pred_c = to_canonical(y_pred, data_format=data_format, axes=axes)

    bct_a = _spatial_mean(y_true_c.astype(np.float64), False)
    bct_b = _spatial_mean(y_pred_c.astype(np.float64), False)

    a = bct_a[batch_index, channel_index]
    b = bct_b[batch_index, channel_index]

    N, M = len(a), len(b)
    w = max(window, abs(N - M)) if window is not None else max(N, M)

    # Build cost matrix and DTW matrix
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, N + 1):
        j_start = max(1, i - w)
        j_end = min(M, i + w)
        for j in range(j_start, j_end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Traceback path
    path = []
    i, j = N, M
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1]])
            if step == 0:
                i -= 1; j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
    path = path[::-1]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # Left: cost matrix + path
    cost_matrix = np.abs(a[:, np.newaxis] - b[np.newaxis, :])
    axs[0].imshow(cost_matrix, origin="lower", aspect="auto", cmap="YlOrRd")
    pi, pj = zip(*path)
    axs[0].plot(pj, pi, color="white", lw=1.5, label="warping path")
    axs[0].set_xlabel("Prediction time step")
    axs[0].set_ylabel("Ground truth time step")
    axs[0].set_title("DTW cost matrix")
    axs[0].legend(fontsize=8)

    # Right: series + alignment lines
    offset = np.max(np.abs(a)) * 2.5
    axs[1].plot(a, color="black", lw=1.6, label="Ground truth")
    axs[1].plot(b - offset, color="#D65F5F", lw=1.6, label=f"Prediction (shifted by {offset:.2g})")
    for i_pt, j_pt in path[::max(1, len(path) // 30)]:
        axs[1].plot([i_pt, j_pt], [a[i_pt], b[j_pt] - offset],
                    color="gray", alpha=0.3, lw=0.7)
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel("Value")
    axs[1].set_title("Series with alignment")
    axs[1].legend(fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    maybe_save(fig, save_path)
    return fig, axs


def plot_channel_correlation_matrix(
    data: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Inter-channel correlation",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Plots the inter-channel correlation matrix as an annotated heatmap.
        Shows how strongly correlated pairs of channels/variables are across
        time and batch, which is useful for comparing whether a model
        preserves multi-variable dependencies.

        Parameters:
        -----------
        - data : np.ndarray
            Input array, any spatial or series shape.
        - channel_names : sequence of str
            Optional labels for each axis tick.
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - ignore_nan : bool
            If True, NaN values are excluded (default: False).
        - title : str
            Plot title.
        - save_path : str
            If given, saves the figure to this path.
        - ax : matplotlib.axes.Axes
            Existing axis to draw on.

        Returns:
        --------
        - (fig, ax) : tuple

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_channel_correlation_matrix

        data = np.random.randn(4, 5, 100)
        fig, ax = plot_channel_correlation_matrix(data, axes="bct",
                                                   channel_names=list("ABCDE"))
        ```
    '''
    from ..metrics.timeseries import cross_channel_correlation

    apply_style()

    corr = cross_channel_correlation(data, data_format=data_format,
                                     axes=axes, ignore_nan=ignore_nan)
    C = corr.shape[0]
    labels = list(channel_names) if channel_names else [f"ch {i}" for i in range(C)]

    if ax is None:
        sz = max(5, C)
        fig, ax = plt.subplots(figsize=(sz, sz))
    else:
        fig = ax.figure

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(C):
        for j in range(C):
            v = corr[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=max(6, 9 - C // 3), color=color)

    maybe_save(fig, save_path)
    return fig, ax
