"""
    Ensemble validation plots, paired with the metrics in `aule.metrics.ensemble`.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .._shapes import to_canonical
from .._guards import requires
from ._style import apply_style, sequential_norm, make_geo_axis, maybe_save, SEQUENTIAL_CMAP

__all__ = ["plot_ensemble_spread_map", "plot_rank_histogram"]


@requires(spatial=True, array_args=("y_ensemble",))
def plot_ensemble_spread_map(
    y_ensemble: np.ndarray,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    channel_index: int = 0,
    time_index: int = 0,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    title: str = "Ensemble spread (1 sigma)",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Map of the ensemble standard deviation at each pixel, for a given
        channel/time slice, averaged over the batch axis if present.

        Parameters:
        -----------
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape).
        - lat, lon : np.ndarray
            Optional 2D coordinate arrays of shape (H, W).
        - channel_index, time_index : int
            Which channel/time slice to display.
        - data_format : str
            "bhwc" or "hwct", required only when each member is 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the spread computation (default: False).
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
        from aule.plots import plot_ensemble_spread_map

        ensemble = np.random.rand(10, 64, 64, 1)
        fig, ax = plot_ensemble_spread_map(ensemble)
        ```
    '''

    apply_style()

    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)  # (M, B, H, W, C, T)

    slice_ = stacked[:, :, :, :, channel_index, time_index]  # (M, B, H, W)

    std_fn = np.nanstd if ignore_nan else np.std
    spread = std_fn(slice_, axis=0)  # (B, H, W)
    spread_field = np.mean(spread, axis=0)  # (H, W)

    use_geo = lat is not None and lon is not None

    if use_geo:
        fig = plt.figure(figsize=(7, 6))
        ax = make_geo_axis(fig)
        import cartopy.crs as ccrs
        im = ax.pcolormesh(lon, lat, spread_field, cmap=SEQUENTIAL_CMAP,
                           norm=sequential_norm(spread_field), transform=ccrs.PlateCarree(), shading="auto")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(np.flipud(spread_field), cmap=SEQUENTIAL_CMAP, norm=sequential_norm(spread_field),
                       interpolation="nearest", aspect="equal")
        ax.axis("off")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    maybe_save(fig, save_path)

    return fig, ax


def plot_rank_histogram(
    counts: np.ndarray,
    title: str = "Rank histogram (Talagrand diagram)",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    '''
        Bar plot of ensemble rank histogram counts, as computed by
        `aule.metrics.rank_histogram`. A flat histogram indicates a
        well-calibrated ensemble; U-shaped indicates under-dispersion;
        dome-shaped indicates over-dispersion.

        Parameters:
        -----------
        - counts : np.ndarray
            1D array of rank counts, as returned by `rank_histogram`.
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
        from aule.metrics import rank_histogram
        from aule.plots import plot_rank_histogram

        gt       = np.random.rand(64, 64, 1)
        ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 64, 64, 1))
        counts = rank_histogram(gt, ensemble)
        fig, ax = plot_rank_histogram(counts)
        ```
    '''

    apply_style()

    ranks = np.arange(len(counts))
    expected = np.full(len(counts), counts.sum() / len(counts))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.bar(ranks, counts, color="#4878CF", alpha=0.85, edgecolor="white", label="Observed")
    ax.plot(ranks, expected, color="black", lw=1.5, ls="--", label="Uniform (well-calibrated)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=9)

    maybe_save(fig, save_path)

    return fig, ax
