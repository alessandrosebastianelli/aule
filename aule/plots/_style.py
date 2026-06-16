"""
    Internal plotting helpers (private module, not part of the public API).

    Provides a clean, consistent default style shared by all aule plots, and
    optional cartopy-based basemaps for spatial plots when lat/lon are given.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Clean default style applied by every aule plot.
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
}

DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = "viridis"
DIFF_CMAP = "bwr"


def apply_style() -> None:
    '''
        Applies the shared aule matplotlib style to the current session.
        Called internally at the start of every plotting function.
    '''

    plt.rcParams.update(STYLE)


def symmetric_norm(data: np.ndarray, pct: float = 2.0) -> mcolors.TwoSlopeNorm:
    '''
        Builds a symmetric (zero-centered) color normalization based on
        a robust percentile range of the data. Useful for difference/bias
        maps where 0 should always sit at the center of the colormap.

        Parameters:
        -----------
        - data : np.ndarray
            Array of values to derive the color range from.
        - pct : float
            Percentile used to clip outliers symmetrically (default: 2.0,
            i.e. uses the 2nd/98th percentile envelope).

        Returns:
        --------
        - norm : matplotlib.colors.TwoSlopeNorm
            Normalization object with vcenter=0.
    '''

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)

    v = float(max(abs(np.percentile(finite, pct)), abs(np.percentile(finite, 100 - pct)), 1e-6))
    return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v)


def sequential_norm(data: np.ndarray, pct: float = 1.0) -> plt.Normalize:
    '''
        Builds a sequential (non zero-centered) color normalization based
        on a robust percentile range of the data.

        Parameters:
        -----------
        - data : np.ndarray
            Array of values to derive the color range from.
        - pct : float
            Percentile used to clip outliers (default: 1.0).

        Returns:
        --------
        - norm : matplotlib.colors.Normalize
            Normalization object spanning [pct, 100-pct] percentiles.
    '''

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return plt.Normalize(vmin=0.0, vmax=1.0)

    lo = float(np.percentile(finite, pct))
    hi = float(np.percentile(finite, 100 - pct))
    if lo == hi:
        hi = lo + 1e-6
    return plt.Normalize(vmin=lo, vmax=hi)


def make_geo_axis(
    fig: plt.Figure,
    rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    subplot_spec=None,
):
    '''
        Creates a matplotlib axis with a cartopy PlateCarree projection,
        already styled with coastlines and borders. Requires cartopy to be
        installed (optional dependency, `pip install aule[geo]`).

        Parameters:
        -----------
        - fig : matplotlib.figure.Figure
            Figure to attach the axis to.
        - rect : tuple
            (left, bottom, width, height) in figure fraction, used when
            subplot_spec is None (default: full figure).
        - subplot_spec : matplotlib.gridspec.SubplotSpec
            If given, the axis is created from this spec instead of rect.

        Returns:
        --------
        - ax : cartopy.mpl.geoaxes.GeoAxes
            Axis with PlateCarree projection, coastlines and borders added.

        Raises:
        -------
        - ImportError
            If cartopy is not installed.
    '''

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError(
            "cartopy is required for geographic basemaps. "
            "Install it with `pip install aule[geo]`."
        ) from exc

    if subplot_spec is not None:
        ax = fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree())
    else:
        ax = fig.add_axes(rect, projection=ccrs.PlateCarree())

    ax.coastlines(resolution="50m", linewidth=0.6, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.3, linestyle="--")

    return ax


def maybe_save(fig: plt.Figure, save_path: Optional[str]) -> None:
    '''
        Saves the figure to disk if a path is given, with tight bounding box.

        Parameters:
        -----------
        - fig : matplotlib.figure.Figure
            Figure to save.
        - save_path : str
            Destination path, or None to skip saving.
    '''

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
