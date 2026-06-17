"""
    General-purpose diagnostic plots: Taylor diagrams, box/violin
    distribution comparisons, and flexible time series.
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import match_shapes, to_canonical
from ._style import apply_style, maybe_save

__all__ = ["plot_taylor_diagram", "plot_boxplot_comparison", "plot_violin_comparison", "plot_time_series", "plot_error_map"]


def plot_taylor_diagram(
    y_true: np.ndarray,
    y_preds: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Optional[Sequence[str]] = None,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Taylor diagram",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Draws a Taylor diagram summarizing, for one or more predictions
        against a single ground truth, the correlation coefficient (angular
        position), the normalized standard deviation (radial distance), and
        implicitly the centered RMSE (distance from the reference point on
        the x-axis). Standard summary plot in climate model evaluation for
        comparing multiple models/predictions at a glance.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_preds : np.ndarray or sequence of np.ndarray
            One prediction array, or a list/tuple of prediction arrays
            (e.g. different models), each the same shape as y_true.
        - labels : sequence of str
            Optional labels for each prediction, shown in the legend.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the statistics (default: False).
        - title : str
            Plot title.
        - save_path : str
            If given, the figure is saved to this path (default: None).
        - ax : matplotlib.axes.Axes
            Existing polar axis to draw on. If None, a new figure/axis is created.

        Returns:
        --------
        - (fig, ax) : tuple
            The matplotlib figure and polar axis, for further customization.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots import plot_taylor_diagram

        gt     = np.random.rand(64, 64, 1)
        pred_a = gt + np.random.normal(0, 0.05, gt.shape)
        pred_b = gt * 1.2 + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_taylor_diagram(gt, [pred_a, pred_b], labels=["Model A", "Model B"])
        ```
    '''

    apply_style()

    if isinstance(y_preds, np.ndarray):
        y_preds = [y_preds]
    if labels is None:
        labels = [f"Prediction {i + 1}" for i in range(len(y_preds))]

    y_true_c = to_canonical(y_true, data_format=data_format)
    a = y_true_c.astype(np.float64).ravel()
    if ignore_nan:
        a_mask = np.isfinite(a)
    std_true = np.std(a[a_mask]) if ignore_nan else np.std(a)

    points = []
    for pred in y_preds:
        _, pred_c = match_shapes(y_true, pred, data_format=data_format)
        b = pred_c.astype(np.float64).ravel()

        if ignore_nan:
            mask = np.isfinite(a) & np.isfinite(b)
            aa, bb = a[mask], b[mask]
        else:
            aa, bb = a, b

        std_pred = np.std(bb)
        r = float(np.corrcoef(aa, bb)[0, 1]) if (aa.size >= 2 and np.std(aa) > 0 and std_pred > 0) else 0.0
        points.append((std_pred / std_true if std_true > 0 else 0.0, r))

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
    else:
        fig = ax.figure

    ax.set_thetamin(0)
    ax.set_thetamax(90)
    max_std = max([p[0] for p in points] + [1.0]) * 1.3

    correlation_ticks = [1.0, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    ax.set_xticks([np.arccos(c) for c in correlation_ticks])
    ax.set_xticklabels([str(c) for c in correlation_ticks])
    ax.set_ylim(0, max_std)
    ax.set_thetagrids([np.degrees(np.arccos(c)) for c in correlation_ticks])

    # reference point: perfect correlation, normalized std = 1
    ax.plot(0, 1.0, marker="*", color="black", markersize=14, label="Reference")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(points), 1)))
    for (std_ratio, r), label, color in zip(points, labels, colors):
        theta = np.arccos(np.clip(r, -1.0, 1.0))
        ax.plot(theta, std_ratio, marker="o", markersize=9, color=color, label=label)

    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax


def plot_boxplot_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    group_by_channel: bool = False,
    title: str = "Distribution comparison (box plot)",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Box plot comparing the distribution of ground truth and prediction
        values, optionally broken down per channel.

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
        - group_by_channel : bool
            If True, draws one pair of boxes per channel instead of pooling
            all channels together (default: False).
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
        from aule.plots import plot_boxplot_comparison

        gt   = np.random.rand(8, 32, 32, 3)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_boxplot_comparison(gt, pred, group_by_channel=True)
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    def _clean(arr: np.ndarray) -> np.ndarray:
        flat = arr.astype(np.float64).ravel()
        return flat[np.isfinite(flat)] if ignore_nan else flat

    if group_by_channel:
        C = y_true_c.shape[3]
        data, positions, colors_list = [], [], []
        for c in range(C):
            data.append(_clean(y_true_c[:, :, :, c, :]))
            data.append(_clean(y_pred_c[:, :, :, c, :]))
            positions.extend([c * 3, c * 3 + 1])
            colors_list.extend(["black", "#D65F5F"])

        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks([c * 3 + 0.5 for c in range(C)])
        ax.set_xticklabels([f"Channel {c}" for c in range(C)])
    else:
        data = [_clean(y_true_c), _clean(y_pred_c)]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.6, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["black", "#D65F5F"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ground truth", "Prediction"])

    ax.set_ylabel("Value")
    ax.set_title(title)

    maybe_save(fig, save_path)

    return fig, ax


def plot_violin_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Distribution comparison (violin plot)",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Violin plot comparing the full distribution shape of ground truth
        and prediction values. An alternative to `plot_boxplot_comparison`
        that also shows distribution density, not just quartiles.

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
        from aule.plots import plot_violin_comparison

        gt   = np.random.rand(8, 32, 32, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_violin_comparison(gt, pred)
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    parts = ax.violinplot([a, b], positions=[0, 1], showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], ["black", "#D65F5F"]):
        body.set_facecolor(color)
        body.set_alpha(0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ground truth", "Prediction"])
    ax.set_ylabel("Value")
    ax.set_title(title)

    maybe_save(fig, save_path)

    return fig, ax


def plot_time_series(
    series: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    x: Optional[np.ndarray] = None,
    confidence_bands: Optional[Sequence[Optional[np.ndarray]]] = None,
    title: str = "Time series comparison",
    xlabel: str = "Time",
    ylabel: str = "Value",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Generic multi-line time series plot, with optional shaded confidence
        bands per series. More flexible than `aule.plots.plot_temporal_trend`
        since it accepts arbitrary precomputed 1D series (e.g. spatial means
        already computed elsewhere, or any other scalar time series) rather
        than requiring a (y_true, y_pred) pair in the standard aule shapes.

        Parameters:
        -----------
        - series : sequence of np.ndarray
            List of 1D arrays to plot, one line per series.
        - labels : sequence of str
            Optional labels for each series, shown in the legend.
        - x : np.ndarray
            Optional shared x-axis values (default: integer sample index).
        - confidence_bands : sequence of np.ndarray or None
            Optional per-series half-widths for a shaded band around each
            line (e.g. standard deviation or confidence interval radius).
            Use None for a given series to skip its band.
        - title : str
            Plot title.
        - xlabel, ylabel : str
            Axis labels.
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
        from aule.plots import plot_time_series

        t = np.arange(100)
        gt_series   = np.sin(t / 10) + np.random.normal(0, 0.05, 100)
        pred_series = np.sin(t / 10) + np.random.normal(0, 0.1, 100)
        fig, ax = plot_time_series(
            [gt_series, pred_series],
            labels=["Ground truth", "Prediction"],
            x=t,
        )
        ```
    '''

    apply_style()

    if labels is None:
        labels = [f"Series {i + 1}" for i in range(len(series))]
    if confidence_bands is None:
        confidence_bands = [None] * len(series)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(series), 1)))

    for s, label, band, color in zip(series, labels, confidence_bands, colors):
        s = np.asarray(s, dtype=np.float64)
        x_vals = np.arange(len(s)) if x is None else x

        ax.plot(x_vals, s, lw=1.8, label=label, color=color)
        if band is not None:
            band = np.asarray(band, dtype=np.float64)
            ax.fill_between(x_vals, s - band, s + band, color=color, alpha=0.15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax


def plot_error_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    batch_index: int = 0,
    channel_index: int = 0,
    time_index: int = 0,
    data_format: Optional[str] = None,
    abs_error: bool = True,
    norm_type: str = "linear",
    norm_kwargs: Optional[dict] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Map of the pointwise error (prediction minus ground truth, or its
        absolute value) for a single (batch, channel, time) slice. Distinct
        from `aule.plots.plot_bias_map`, which shows the mean signed bias
        averaged over batch and time rather than a single error snapshot.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - lat, lon : np.ndarray
            Optional 2D coordinate arrays of shape (H, W). When given, the
            map is drawn with a cartopy PlateCarree basemap (requires `aule[geo]`).
        - batch_index, channel_index, time_index : int
            Which slice to display, when the corresponding axis has size > 1.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - abs_error : bool
            If True (default), shows |pred - true| with a sequential colormap;
            if False, shows the signed error with a diverging colormap.
        - norm_type : str
            Color normalization, used only when `abs_error=False` (the
            signed/diverging branch): "linear" (default), "power"
            (emphasizes extremes via `gamma` in `norm_kwargs`), "symlog"
            (strongest extreme contrast via `linthresh` in `norm_kwargs`),
            or "twoslope" (off-zero `vcenter` in `norm_kwargs`). See
            `aule.plots._style.resolve_diverging_norm`.
        - norm_kwargs : dict
            Extra keyword arguments forwarded to the chosen normalization.
        - title : str
            Plot title. Defaults to a description based on `abs_error`.
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
        from aule.plots import plot_error_map

        gt   = np.random.rand(64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_error_map(gt, pred)

        # emphasize extreme signed errors
        fig, ax = plot_error_map(gt, pred, abs_error=False, norm_type="symlog",
                                  norm_kwargs={"linthresh": 0.02})
        ```
    '''

    from ._style import sequential_norm, make_geo_axis, resolve_diverging_norm, SEQUENTIAL_CMAP, DIFF_CMAP

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    gt_field = y_true_c[batch_index, :, :, channel_index, time_index].astype(np.float64)
    pred_field = y_pred_c[batch_index, :, :, channel_index, time_index].astype(np.float64)

    error_field = pred_field - gt_field
    if abs_error:
        error_field = np.abs(error_field)
        cmap = SEQUENTIAL_CMAP
        norm = sequential_norm(error_field)
        default_title = "Absolute error map (|pred - GT|)"
    else:
        cmap = DIFF_CMAP
        norm = resolve_diverging_norm(error_field, norm_type=norm_type, **(norm_kwargs or {}))
        default_title = "Signed error map (pred - GT)"

    use_geo = lat is not None and lon is not None

    if use_geo:
        fig = plt.figure(figsize=(7, 6))
        ax = make_geo_axis(fig)
        import cartopy.crs as ccrs
        im = ax.pcolormesh(lon, lat, error_field, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(np.flipud(error_field), cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        ax.axis("off")

    ax.set_title(title or default_title)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    maybe_save(fig, save_path)

    return fig, ax
