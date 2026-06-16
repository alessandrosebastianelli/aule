"""
    Core distribution and scatter plots for comparing ground truth and prediction.

    Every plotting function returns the (fig, ax) matplotlib objects, so further
    customization is always possible after calling the function, in addition to
    optional automatic saving via `save_path`.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import match_shapes, finite_mask
from ._style import apply_style, maybe_save

__all__ = ["plot_scatter", "plot_qq", "plot_histogram_comparison", "plot_error_histogram"]


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    max_points: int = 5000,
    title: str = "Scatter: ground truth vs prediction",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Scatter plot of ground truth vs prediction values, with a 1:1
        reference line, randomly subsampled for readability when the
        number of points is large.

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
        - max_points : int
            Maximum number of points to draw (default: 5000); the rest are
            randomly subsampled for a responsive plot.
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
        from aule.plots import plot_scatter

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_scatter(gt, pred, save_path="scatter.png")
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]

    rng = np.random.default_rng(42)
    if a.size > max_points:
        idx = rng.choice(a.size, max_points, replace=False)
        a, b = a[idx], b[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(a, b, s=8, alpha=0.35, color="#4878CF", edgecolor="none")

    lims = [float(min(a.min(), b.min())), float(max(a.max(), b.max()))] if a.size else [0, 1]
    ax.plot(lims, lims, "r--", lw=1.2, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    maybe_save(fig, save_path)

    return fig, ax


def plot_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    n_quantiles: int = 2000,
    title: str = "Q-Q plot: ground truth vs prediction",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Quantile-quantile plot comparing the marginal distributions of
        ground truth and prediction.

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
        - n_quantiles : int
            Number of quantile pairs to compute and plot (default: 2000).
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
        from aule.plots import plot_qq

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_qq(gt, pred)
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

    quantiles = np.linspace(0, 100, n_quantiles)
    q_true = np.percentile(a, quantiles)
    q_pred = np.percentile(b, quantiles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(q_true, q_pred, s=6, alpha=0.5, color="#4878CF")
    lims = [float(min(q_true.min(), q_pred.min())), float(max(q_true.max(), q_pred.max()))]
    ax.plot(lims, lims, "r--", lw=1.2, label="1:1")
    ax.set_xlabel("Ground truth quantiles")
    ax.set_ylabel("Prediction quantiles")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    maybe_save(fig, save_path)

    return fig, ax


def plot_histogram_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    bins: int = 60,
    title: str = "Distribution comparison",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Overlaid histograms of ground truth and prediction values, sharing
        the same bin edges for a direct visual comparison.

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
        - bins : int
            Number of histogram bins (default: 60).
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
        from aule.plots import plot_histogram_comparison

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_histogram_comparison(gt, pred)
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

    shared_bins = np.linspace(
        min(a.min(), b.min()) if a.size and b.size else 0,
        max(a.max(), b.max()) if a.size and b.size else 1,
        bins,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.hist(a, bins=shared_bins, density=True, histtype="step", lw=2, color="black", label="Ground truth")
    ax.hist(b, bins=shared_bins, density=True, histtype="step", lw=2, ls="--", color="#D65F5F", label="Prediction")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax


def plot_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    bins: int = 60,
    title: str = "Error distribution",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Histogram of the pixel-wise error (prediction minus ground truth).

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
        - bins : int
            Number of histogram bins (default: 60).
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
        from aule.plots import plot_error_histogram

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        fig, ax = plot_error_histogram(gt, pred)
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    diff = (y_pred_c.astype(np.float64) - y_true_c.astype(np.float64)).ravel()

    if ignore_nan:
        diff = diff[np.isfinite(diff)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    ax.hist(diff, bins=bins, color="#D65F5F", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Prediction - Ground truth")
    ax.set_ylabel("Count")
    ax.set_title(title)

    maybe_save(fig, save_path)

    return fig, ax
