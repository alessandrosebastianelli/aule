"""
    Advanced spatio-temporal and distributional plots: Hovmoller diagrams,
    CDF comparisons, radial spectral density profiles, and multi-panel
    time evolution snapshots.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import match_shapes, to_canonical
from ._style import apply_style, maybe_save, sequential_norm, SEQUENTIAL_CMAP

__all__ = ["plot_hovmoller", "plot_cdf_comparison", "plot_spectral_density", "plot_time_evolution"]


def plot_hovmoller(
    data: np.ndarray,
    axis: str = "lat",
    batch_index: int = 0,
    channel_index: int = 0,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Hovmoller diagram",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Draws a Hovmoller diagram: a 2D space-time plot with time on one
        axis and a single spatial dimension (latitude or longitude, after
        averaging over the other spatial dimension) on the other, colored
        by value. A standard visualization in climate science for spotting
        propagating features (e.g. traveling waves, monsoon progression,
        cold front movement) that a single spatial snapshot would miss.

        Parameters:
        -----------
        - data : np.ndarray
            Array with a time axis (shapes (b) or (d)).
        - axis : str
            Which spatial axis to keep on the y-axis: "lat" averages over
            longitude (W) and keeps H as the spatial axis; "lon" averages
            over latitude (H) and keeps W as the spatial axis (default: "lat").
        - batch_index, channel_index : int
            Which batch/channel slice to use, when the corresponding axis has size > 1.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Must be
            "hwct" here, since a time axis is required.
        - ignore_nan : bool
            If True, non-finite values are excluded from the averaging (default: False).
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
        from aule.plots import plot_hovmoller

        H, W, T = 48, 48, 60
        lat_idx = np.arange(H).reshape(H, 1, 1, 1)
        t_idx = np.arange(T).reshape(1, 1, 1, T)
        # a feature that propagates in latitude over time
        data = np.sin(2 * np.pi * (lat_idx / H - t_idx / T)) [..., 0:1, :]
        data = np.broadcast_to(data, (H, W, 1, T)).copy()
        fig, ax = plot_hovmoller(data, axis="lat", data_format="hwct")
        ```
    '''

    if axis not in ("lat", "lon"):
        raise ValueError("axis must be 'lat' or 'lon'")

    apply_style()

    data_c = to_canonical(data, data_format=data_format)

    field = data_c[batch_index, :, :, channel_index, :].astype(np.float64)  # (H, W, T)

    mean_fn = np.nanmean if ignore_nan else np.mean
    if axis == "lat":
        profile = mean_fn(field, axis=1)  # average over W -> (H, T)
        ylabel = "Latitude index (H)"
    else:
        profile = mean_fn(field, axis=0)  # average over H -> (W, T)
        ylabel = "Longitude index (W)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    norm = sequential_norm(profile)
    im = ax.imshow(profile, aspect="auto", origin="lower", cmap=SEQUENTIAL_CMAP, norm=norm,
                   extent=[0, profile.shape[1], 0, profile.shape[0]])

    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    maybe_save(fig, save_path)

    return fig, ax


def plot_cdf_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "CDF comparison",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Plots the empirical cumulative distribution functions (CDFs) of
        ground truth and prediction overlaid on the same axis. Complements
        `aule.plots.plot_qq`: where the Q-Q plot shows quantile-by-quantile
        agreement against a 1:1 line, this shows the actual distribution
        shapes directly, making it easier to spot where in the value range
        a model over- or under-represents probability mass (e.g. relevant
        when diagnosing what `aule.metrics.quantile_mapping_bias` measures
        numerically).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded before plotting (default: False).
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
        from aule.plots import plot_cdf_comparison

        gt   = np.random.exponential(1.0, (8, 64, 64, 1))
        pred = gt * 1.1
        fig, ax = plot_cdf_comparison(gt, pred)
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)

    a_cdf = np.arange(1, len(a_sorted) + 1) / len(a_sorted)
    b_cdf = np.arange(1, len(b_sorted) + 1) / len(b_sorted)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.plot(a_sorted, a_cdf, color="black", lw=1.8, label="Ground truth")
    ax.plot(b_sorted, b_cdf, color="#D65F5F", lw=1.8, ls="--", label="Prediction")

    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(title)
    ax.legend(fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax


def plot_spectral_density(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batch_index: int = 0,
    channel_index: int = 0,
    time_index: int = 0,
    data_format: Optional[str] = None,
    title: str = "Radially-averaged power spectral density",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Plots the radially-averaged power spectral density (the same 1D
        profile compared numerically by `aule.metrics.psd_radial_error`)
        of ground truth and prediction overlaid on a log-log axis. Useful
        for visually identifying at which spatial frequencies a model loses
        or adds power (e.g. excessive smoothing at high frequencies,
        checkerboard artifacts as spikes at specific frequencies).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - batch_index, channel_index, time_index : int
            Which slice to evaluate, when the corresponding axis has size > 1.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
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
        from aule.plots import plot_spectral_density

        gt   = np.random.rand(64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        fig, ax = plot_spectral_density(gt, pred)
        ```
    '''

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    true_field = y_true_c[batch_index, :, :, channel_index, time_index].astype(np.float64)
    pred_field = y_pred_c[batch_index, :, :, channel_index, time_index].astype(np.float64)

    H, W = true_field.shape

    true_psd = np.abs(np.fft.rfft2(true_field)) / (H * W)
    pred_psd = np.abs(np.fft.rfft2(pred_field)) / (H * W)

    def _radial_average(psd_2d: np.ndarray) -> np.ndarray:
        h, w = psd_2d.shape
        yy, xx = np.indices((h, w))
        r_int = np.sqrt(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2).astype(np.int64)
        max_r = r_int.max()
        profile = np.zeros(max_r + 1, dtype=np.float64)
        counts = np.zeros(max_r + 1, dtype=np.float64)
        np.add.at(profile, r_int.ravel(), psd_2d.ravel())
        np.add.at(counts, r_int.ravel(), 1.0)
        counts[counts == 0] = 1.0
        return profile / counts

    true_profile = _radial_average(true_psd)
    pred_profile = _radial_average(pred_psd)

    n = min(len(true_profile), len(pred_profile))
    freqs = np.arange(1, n)  # skip the zero-frequency (DC) bin for the log plot

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.loglog(freqs, true_profile[1:n], color="black", lw=1.8, label="Ground truth")
    ax.loglog(freqs, pred_profile[1:n], color="#D65F5F", lw=1.8, ls="--", label="Prediction")

    ax.set_xlabel("Radial spatial frequency (pixels$^{-1}$)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.legend(fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax


def plot_time_evolution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_indices: Optional[Tuple[int, ...]] = None,
    batch_index: int = 0,
    channel_index: int = 0,
    data_format: Optional[str] = None,
    title: str = "Time evolution",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    '''
        Draws a multi-panel snapshot of ground truth (top row) and
        prediction (bottom row) at several time steps, sharing a common
        color scale. Useful for visually inspecting how forecast quality
        evolves over a sequence, complementing the single-slice
        `aule.plots.plot_field_comparison`.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, with a time axis (shapes (b) or (d)).
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - time_indices : tuple of int
            Which time steps to display. If None, picks up to 5 evenly
            spaced steps across the available time axis (default: None).
        - batch_index, channel_index : int
            Which batch/channel slice to use, when the corresponding axis has size > 1.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Must be
            "hwct" here, since a time axis is required.
        - title : str
            Figure-level title.
        - save_path : str
            If given, the figure is saved to this path (default: None).

        Returns:
        --------
        - (fig, axes) : tuple
            The matplotlib figure and a 2D array of axes (2 rows: ground
            truth, prediction; one column per time step), for further customization.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_time_evolution

        gt   = np.random.rand(32, 32, 1, 20)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, axes = plot_time_evolution(gt, pred, data_format="hwct")
        ```
    '''

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    T = y_true_c.shape[-1]
    if time_indices is None:
        n_panels = min(5, T)
        time_indices = tuple(np.linspace(0, T - 1, n_panels).astype(int))

    n_cols = len(time_indices)

    true_fields = [y_true_c[batch_index, :, :, channel_index, t].astype(np.float64) for t in time_indices]
    pred_fields = [y_pred_c[batch_index, :, :, channel_index, t].astype(np.float64) for t in time_indices]

    all_values = np.concatenate([f.ravel() for f in true_fields + pred_fields])
    norm = sequential_norm(all_values)

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, t in enumerate(time_indices):
        im = axes[0, col].imshow(np.flipud(true_fields[col]), cmap=SEQUENTIAL_CMAP, norm=norm,
                                  interpolation="nearest", aspect="equal")
        axes[0, col].axis("off")
        axes[0, col].set_title(f"t={t}", fontsize=10)

        axes[1, col].imshow(np.flipud(pred_fields[col]), cmap=SEQUENTIAL_CMAP, norm=norm,
                             interpolation="nearest", aspect="equal")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Ground truth", fontsize=10)
    axes[1, 0].set_ylabel("Prediction", fontsize=10)
    for row in range(2):
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    fig.suptitle(title, fontsize=12)

    maybe_save(fig, save_path)

    return fig, axes
