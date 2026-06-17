"""
    Classification and probabilistic-calibration plots, paired with the
    metrics in `aule.metrics.classification` and `aule.metrics.uncertainty`.
"""

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import match_shapes, to_canonical
from ._style import apply_style, maybe_save

__all__ = ["plot_confusion_matrix", "plot_reliability_diagram"]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    data_format: Optional[str] = None,
    title: str = "Confusion matrix",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Plots the confusion matrix between ground truth and predicted
        label maps (binary or multi-class), as a heatmap with annotated cell values.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth label map, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Predicted label map, same shape as y_true.
        - num_classes : int
            Number of classes. If None, inferred from the unique values
            present in y_true/y_pred.
        - class_names : sequence of str
            Optional display names for each class, in class-index order.
        - normalize : bool
            If True (default), each row is normalized to sum to 1 (i.e.
            shows the fraction of each true class predicted as each class);
            if False, shows raw pixel counts.
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
        from aule.plots import plot_confusion_matrix

        gt   = np.random.randint(0, 4, (64, 64, 1))
        pred = np.random.randint(0, 4, (64, 64, 1))
        fig, ax = plot_confusion_matrix(gt, pred, num_classes=4,
                                         class_names=["water", "forest", "urban", "bare soil"])
        ```
    '''

    apply_style()

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.unique(np.concatenate([y_true_c.ravel(), y_pred_c.ravel()]))

    n_classes = len(classes)
    sorted_classes = np.sort(classes)

    a_flat = y_true_c.ravel()
    b_flat = y_pred_c.ravel()

    true_idx = np.searchsorted(sorted_classes, a_flat)
    pred_idx = np.searchsorted(sorted_classes, b_flat)
    valid = (true_idx < n_classes) & (pred_idx < n_classes)

    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(confusion, (true_idx[valid], pred_idx[valid]), 1.0)

    display = confusion.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = display / row_sums

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes)))
    else:
        fig = ax.figure

    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalize else display.max())
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = class_names if class_names is not None else [str(c) for c in sorted_classes]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    fmt = "{:.2f}" if normalize else "{:.0f}"
    for i in range(n_classes):
        for j in range(n_classes):
            value = display[i, j]
            color = "white" if value > (0.5 if normalize else display.max() / 2) else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center", color=color, fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_ensemble: np.ndarray,
    threshold: float,
    n_bins: int = 10,
    data_format: Optional[str] = None,
    title: str = "Reliability diagram",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Plots a reliability (calibration) diagram for the binary event
        "value exceeds threshold": forecast probability (from the ensemble
        fraction exceeding the threshold) bin-averaged against the observed
        frequency of the event in that bin. A perfectly calibrated forecast
        lies on the diagonal. Pairs with `aule.metrics.brier_score`.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes (no ensemble axis).
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape).
        - threshold : float
            Threshold defining the binary event (value > threshold).
        - n_bins : int
            Number of probability bins over [0, 1] (default: 10).
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
        from aule.plots import plot_reliability_diagram

        gt       = np.random.rand(64, 64, 1)
        ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 64, 64, 1))
        fig, ax = plot_reliability_diagram(gt, ensemble, threshold=0.7)
        ```
    '''

    apply_style()

    y_true_c = to_canonical(y_true, data_format=data_format)
    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)

    observed = (y_true_c.astype(np.float64) > threshold).astype(np.float64).ravel()
    forecast_prob = np.mean((stacked > threshold).astype(np.float64), axis=0).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(forecast_prob, bin_edges) - 1, 0, n_bins - 1)

    bin_mean_forecast = np.full(n_bins, np.nan)
    bin_mean_observed = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = bin_idx == b
        bin_counts[b] = np.sum(mask)
        if bin_counts[b] > 0:
            bin_mean_forecast[b] = np.mean(forecast_prob[mask])
            bin_mean_observed[b] = np.mean(observed[mask])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfectly calibrated")

    valid = bin_counts > 0
    ax.plot(bin_mean_forecast[valid], bin_mean_observed[valid], marker="o", color="#4878CF", label="Forecast")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    maybe_save(fig, save_path)

    return fig, ax
