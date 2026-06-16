"""
    Spatial map plots for ground truth, prediction and their differences.

    All functions accept optional `lat`/`lon` 2D coordinate arrays. When given,
    maps are drawn on a clean cartopy PlateCarree basemap with coastlines and
    borders (requires the optional `aule[geo]` dependency). When omitted, maps
    fall back to a plain `imshow` with no geographic projection.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import to_canonical
from ._style import apply_style, symmetric_norm, sequential_norm, make_geo_axis, maybe_save, DIVERGING_CMAP, DIFF_CMAP

__all__ = ["plot_field_comparison", "plot_bias_map", "plot_correlation_map"]


def _style_geo_axis(ax: plt.Axes) -> None:
    '''
        Applies coastlines, borders and a light gridline style to an
        already-created cartopy GeoAxes.

        Parameters:
        -----------
        - ax : cartopy.mpl.geoaxes.GeoAxes
            Axis to style in place.
    '''

    import cartopy.feature as cfeature

    ax.coastlines(resolution="50m", linewidth=0.6, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.3, linestyle="--")


def _draw_field(
    fig: plt.Figure,
    ax: plt.Axes,
    field: np.ndarray,
    lat: Optional[np.ndarray],
    lon: Optional[np.ndarray],
    cmap: str,
    norm,
    title: str,
):
    '''
        Draws a single 2D field on the given axis, using a cartopy basemap
        if lat/lon are provided, otherwise a plain imshow.

        Parameters:
        -----------
        - fig : matplotlib.figure.Figure
            Parent figure (used for the colorbar).
        - ax : matplotlib.axes.Axes
            Axis to draw on (must already be a GeoAxes if lat/lon are given).
        - field : np.ndarray
            2D array (H, W) to display.
        - lat, lon : np.ndarray
            Optional 2D coordinate arrays, same shape as field.
        - cmap : str
            Colormap name.
        - norm : matplotlib normalization object.
        - title : str
            Axis title.

        Returns:
        --------
        - im : the image/mesh artist, for further customization.
    '''

    if lat is not None and lon is not None:
        import cartopy.crs as ccrs

        im = ax.pcolormesh(lon, lat, field, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
    else:
        im = ax.imshow(np.flipud(field), cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        ax.axis("off")

    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    return im


def plot_field_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    batch_index: int = 0,
    channel_index: int = 0,
    time_index: int = 0,
    data_format: Optional[str] = None,
    title: str = "Ground truth vs prediction",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    '''
        Side-by-side maps of ground truth, prediction and their difference,
        for a single (batch, channel, time) slice.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - lat, lon : np.ndarray
            Optional 2D coordinate arrays of shape (H, W). When given, maps
            are drawn with a cartopy PlateCarree basemap (requires `aule[geo]`).
        - batch_index, channel_index, time_index : int
            Which slice to display, when the corresponding axis has size > 1.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - title : str
            Figure-level title.
        - save_path : str
            If given, the figure is saved to this path (default: None).

        Returns:
        --------
        - (fig, axes) : tuple
            The matplotlib figure and an array of 3 axes
            (ground truth, prediction, difference), for further customization.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_field_comparison

        gt   = np.random.rand(64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, axes = plot_field_comparison(gt, pred)
        ```
    '''

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    gt_field = y_true_c[batch_index, :, :, channel_index, time_index].astype(np.float64)
    pred_field = y_pred_c[batch_index, :, :, channel_index, time_index].astype(np.float64)
    diff_field = pred_field - gt_field

    use_geo = lat is not None and lon is not None

    if use_geo:
        import cartopy.crs as ccrs
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": ccrs.PlateCarree()})
        for axis in axes:
            _style_geo_axis(axis)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    field_norm = sequential_norm(gt_field)
    diff_norm = symmetric_norm(diff_field)

    _draw_field(fig, axes[0], gt_field, lat, lon, DIVERGING_CMAP, field_norm, "Ground truth")
    _draw_field(fig, axes[1], pred_field, lat, lon, DIVERGING_CMAP, field_norm, "Prediction")
    _draw_field(fig, axes[2], diff_field, lat, lon, DIFF_CMAP, diff_norm, "Difference (pred - GT)")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    maybe_save(fig, save_path)

    return fig, axes


def plot_bias_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    channel_index: int = 0,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Spatial bias map",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Map of the mean bias (prediction minus ground truth) at each pixel,
        averaged over batch and time.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - lat, lon : np.ndarray
            Optional 2D coordinate arrays of shape (H, W).
        - channel_index : int
            Which channel to display, when C > 1.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the mean (default: False).
        - title : str
            Plot title.
        - save_path : str
            If given, the figure is saved to this path (default: None).

        Returns:
        --------
        - (fig, ax) : tuple
            The matplotlib figure and axis, for further customization.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_bias_map

        gt   = np.random.rand(20, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_bias_map(gt, pred)
        ```
    '''

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    diff = (y_pred_c[:, :, :, channel_index, :].astype(np.float64) - y_true_c[:, :, :, channel_index, :].astype(np.float64))

    if ignore_nan:
        bias_field = np.nanmean(np.where(np.isfinite(diff), diff, np.nan), axis=(0, 3))
    else:
        bias_field = np.mean(diff, axis=(0, 3))

    use_geo = lat is not None and lon is not None

    if use_geo:
        fig = plt.figure(figsize=(7, 6))
        ax = make_geo_axis(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))

    norm = symmetric_norm(bias_field)
    _draw_field(fig, ax, bias_field, lat, lon, DIFF_CMAP, norm, title)

    maybe_save(fig, save_path)

    return fig, ax


def plot_correlation_map(
    r_map: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    channel_index: int = 0,
    title: str = "Pixel-wise temporal correlation",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Map of pixel-wise Pearson correlation, as computed by
        `aule.metrics.pixelwise_temporal_correlation`.

        Parameters:
        -----------
        - r_map : np.ndarray
            Correlation map of shape (H, W, C), as returned by
            `pixelwise_temporal_correlation`.
        - lat, lon : np.ndarray
            Optional 2D coordinate arrays of shape (H, W).
        - channel_index : int
            Which channel to display, when C > 1.
        - title : str
            Plot title.
        - save_path : str
            If given, the figure is saved to this path (default: None).

        Returns:
        --------
        - (fig, ax) : tuple
            The matplotlib figure and axis, for further customization.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import pixelwise_temporal_correlation
        from aule.plots import plot_correlation_map

        gt   = np.random.rand(50, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        r_map = pixelwise_temporal_correlation(gt, pred)
        fig, ax = plot_correlation_map(r_map)
        ```
    '''

    apply_style()

    field = r_map[:, :, channel_index].astype(np.float64)

    use_geo = lat is not None and lon is not None

    if use_geo:
        fig = plt.figure(figsize=(7, 6))
        ax = make_geo_axis(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))

    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    _draw_field(fig, ax, field, lat, lon, "RdYlGn", norm, title)

    maybe_save(fig, save_path)

    return fig, ax
