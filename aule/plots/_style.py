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


def power_norm(
    data: Optional[np.ndarray] = None,
    gamma: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> mcolors.FuncNorm:
    '''
        Builds a symmetric power-law color normalization: values are
        transformed via `sign(x) * |x|**gamma` before mapping to the
        colormap. Useful for divergent data dominated by near-zero noise,
        where a small `gamma` (e.g. 0.3-0.5) pulls more of the colormap
        towards extreme values so they stand out more than under a plain
        linear normalization, while keeping the transform smooth and
        continuous everywhere (no break in the color gradient near zero).

        Parameters:
        -----------
        - data : np.ndarray
            Optional array to derive a symmetric (vmin, vmax) range from,
            via the max absolute finite value. Ignored if both vmin and
            vmax are given explicitly.
        - gamma : float
            Power-law exponent, in (0, 1]. Smaller values compress the
            near-zero range more aggressively, pushing color contrast
            towards the extremes (default: 0.5, a square-root normalization).
            Values much below ~0.3 tend to make background noise visible
            as speckle; tune gradually.
        - vmin, vmax : float
            Explicit symmetric range. If not given, both default to the
            max absolute finite value of `data` (or +-1.0 if `data` is
            also not given).

        Returns:
        --------
        - norm : matplotlib.colors.FuncNorm
            Normalization object usable as the `norm=` argument to
            `imshow`/`pcolormesh`.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots._style import power_norm

        data = np.random.normal(0, 0.05, (100, 100))
        norm = power_norm(data, gamma=0.4)
        ```
    '''

    if vmin is None or vmax is None:
        if data is not None:
            finite = data[np.isfinite(data)]
            v = float(np.max(np.abs(finite))) if finite.size > 0 else 1.0
        else:
            v = 1.0
        vmin = -v if vmin is None else vmin
        vmax = v if vmax is None else vmax

    def forward(x):
        return np.sign(x) * np.abs(x) ** gamma

    def inverse(y):
        return np.sign(y) * np.abs(y) ** (1.0 / gamma)

    return mcolors.FuncNorm((forward, inverse), vmin=vmin, vmax=vmax)


def symlog_norm(
    data: Optional[np.ndarray] = None,
    linthresh: Optional[float] = None,
    linscale: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    base: float = 10.0,
) -> mcolors.SymLogNorm:
    '''
        Builds a symmetric-log color normalization: linear within
        +-`linthresh` of zero (so background noise near zero stays calm),
        logarithmic beyond it (so extreme values spread across more of the
        colormap, instead of saturating quickly to the same dark color as
        under a linear normalization). Generally gives the strongest
        extreme-value contrast among aule's normalization helpers, at the
        cost of making small near-threshold fluctuations more visible as
        speckle outside the linear zone.

        Parameters:
        -----------
        - data : np.ndarray
            Optional array used to pick sensible defaults for `linthresh`
            (a small fraction of the data's robust spread) and the
            symmetric (vmin, vmax) range, when not given explicitly.
        - linthresh : float
            Half-width of the linear region around zero. If None and
            `data` is given, defaults to the 75th percentile of |data|
            (a reasonable "most values are noise" cutoff); if `data` is
            also None, defaults to 0.03.
        - linscale : float
            Relative size of the linear region compared to one decade of
            the log region, as in `matplotlib.colors.SymLogNorm` (default: 0.5).
        - vmin, vmax : float
            Explicit symmetric range. Defaults to the max absolute finite
            value of `data` (or +-1.0 if `data` is also not given).
        - base : float
            Logarithm base for the non-linear region (default: 10).

        Returns:
        --------
        - norm : matplotlib.colors.SymLogNorm
            Normalization object usable as the `norm=` argument to
            `imshow`/`pcolormesh`.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots._style import symlog_norm

        data = np.random.normal(0, 0.05, (100, 100))
        norm = symlog_norm(data, linthresh=0.03)
        ```
    '''

    if data is not None:
        finite = data[np.isfinite(data)]
    else:
        finite = np.array([])

    if vmin is None or vmax is None:
        v = float(np.max(np.abs(finite))) if finite.size > 0 else 1.0
        vmin = -v if vmin is None else vmin
        vmax = v if vmax is None else vmax

    if linthresh is None:
        if finite.size > 0:
            linthresh = float(np.percentile(np.abs(finite), 75))
            if linthresh <= 0:
                linthresh = 0.03
        else:
            linthresh = 0.03

    return mcolors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax, base=base)


def asymmetric_twoslope_norm(
    data: Optional[np.ndarray] = None,
    vcenter: float = 0.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> mcolors.TwoSlopeNorm:
    '''
        Builds a two-slope color normalization anchoring a chosen center
        value (not necessarily 0) to the colormap's midpoint, with
        independent linear scaling on either side. Useful for divergent
        data that is not symmetric around zero (e.g. mostly small positive
        anomalies with a few large negative ones, or vice versa), where a
        plain symmetric normalization would waste colormap range on one
        side and clip the other.

        Parameters:
        -----------
        - data : np.ndarray
            Optional array to derive (vmin, vmax) from (min/max finite
            values), when not given explicitly.
        - vcenter : float
            The data value mapped to the colormap's midpoint color
            (default: 0.0).
        - vmin, vmax : float
            Explicit range. Defaults to the min/max finite value of `data`
            (or (-1.0, 1.0) if `data` is also not given). Both are nudged
            outward slightly if they would otherwise equal `vcenter`.

        Returns:
        --------
        - norm : matplotlib.colors.TwoSlopeNorm
            Normalization object usable as the `norm=` argument to
            `imshow`/`pcolormesh`.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots._style import asymmetric_twoslope_norm

        # mostly small positive values with a few strong negative dips
        data = np.random.exponential(0.2, (100, 100)) - 0.3
        norm = asymmetric_twoslope_norm(data, vcenter=0.0)
        ```
    '''

    if vmin is None or vmax is None:
        if data is not None:
            finite = data[np.isfinite(data)]
        else:
            finite = np.array([])
        if finite.size > 0:
            vmin = float(np.min(finite)) if vmin is None else vmin
            vmax = float(np.max(finite)) if vmax is None else vmax
        else:
            vmin = -1.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

    if vmin >= vcenter:
        vmin = vcenter - 1e-6
    if vmax <= vcenter:
        vmax = vcenter + 1e-6

    return mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)


def resolve_diverging_norm(
    data: Optional[np.ndarray] = None,
    norm_type: str = "linear",
    pct: float = 2.0,
    **norm_kwargs,
):
    '''
        Dispatches to the appropriate divergent color normalization
        helper by name, so plotting functions can expose a single
        `norm_type` string parameter instead of requiring the caller to
        import and build a normalization object directly.

        Parameters:
        -----------
        - data : np.ndarray
            Array of values to derive the normalization range from.
        - norm_type : str
            One of:
            - "linear" (default): plain symmetric linear scaling via
              `symmetric_norm`, using the `pct` percentile envelope.
            - "power": power-law normalization via `power_norm`
              (pass `gamma` in `norm_kwargs` to tune, default 0.5).
            - "symlog": symmetric-log normalization via `symlog_norm`
              (pass `linthresh`/`linscale`/`base` in `norm_kwargs`).
            - "twoslope": two-slope normalization via
              `asymmetric_twoslope_norm` (pass `vcenter` in `norm_kwargs`
              for asymmetric data; defaults to 0.0).
        - pct : float
            Percentile envelope used only by `norm_type="linear"` (default: 2.0).
        - **norm_kwargs
            Forwarded to the underlying normalization helper.

        Returns:
        --------
        - norm : a matplotlib Normalize-compatible object.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.plots._style import resolve_diverging_norm

        data = np.random.normal(0, 0.05, (100, 100))
        norm = resolve_diverging_norm(data, norm_type="symlog", linthresh=0.03)
        ```
    '''

    if norm_type == "linear":
        return symmetric_norm(data, pct=pct)
    if norm_type == "power":
        return power_norm(data, **norm_kwargs)
    if norm_type == "symlog":
        return symlog_norm(data, **norm_kwargs)
    if norm_type == "twoslope":
        return asymmetric_twoslope_norm(data, **norm_kwargs)

    raise ValueError(
        f"Unknown norm_type '{norm_type}'. Expected one of: 'linear', 'power', 'symlog', 'twoslope'."
    )


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
