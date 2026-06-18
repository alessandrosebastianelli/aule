"""
    Internal shape-handling utilities (private module, not part of the public API).

    aule accepts numpy arrays in one of four shapes:

        (a) (batch, H, W, C)
        (b) (batch, H, W, C, T)
        (c) (H, W, C)
        (d) (H, W, C, T)

    Every public function (metric or plot) normalizes its inputs to a single
    canonical 5D shape (batch, H, W, C, T) before doing any computation, so the
    actual metric/plot logic only has to be written once.
"""

from typing import Optional, Tuple
import numpy as np


"""
    Internal shape-handling utilities (private module, not part of the public API).

    aule normalizes every input array to a single canonical 6D shape
    (batch, H, W, C, T) before any computation, so metric/plot logic is
    only written once. Two input families are supported:

    Spatial family (the original 4 shapes, disambiguated via `data_format`
    when needed):

        (a) (batch, H, W, C)
        (b) (batch, H, W, C, T)
        (c) (H, W, C)
        (d) (H, W, C, T)

    Series family (no spatial extent - pure time series, optionally with a
    batch and/or channel axis), disambiguated via the explicit `axes`
    parameter since dimensionality alone is ambiguous (e.g. a 1D array
    could be a single time series or a single channel's worth of
    something else):

        (e) (T,)            axes="t"
        (f) (C,)            axes="c"
        (g) (B, T)           axes="bt"
        (h) (C, T)           axes="ct"
        (i) (B, C, T)         axes="bct"

    Series inputs are promoted to the same canonical (batch, H, W, C, T)
    shape with H=W=1 (and T=1 too, for the "c"-only case): a pure time
    series is simply a spatial field degenerated to a single point. This
    means metrics/plots that only care about value comparison (rmse, mae,
    pearson_r, plot_scatter, ...) work transparently on series inputs with
    no extra code. Functions that genuinely need spatial or temporal
    extent (gradient_error, ssim, plot_field_comparison, ...) instead
    declare their requirements via `aule._guards.requires` and reject
    degenerate inputs by default, unless explicitly forced - see that
    module for details.
"""

from typing import Optional, Tuple
import numpy as np

from ._logging import logger

# Which canonical axes are real data vs synthetic size-1 padding, tracked
# as a small marker object attached to canonical arrays' provenance. We
# can't literally attach metadata to a plain ndarray, so `to_canonical`
# also returns this alongside the array; callers that don't care (the
# vast majority) just ignore it via `to_canonical_array`-style helpers.
_VALID_AXES = ("b", "h", "w", "c", "t")


def _axes_from_data_format(ndim: int, data_format: Optional[str]) -> str:
    '''
        Maps the legacy `data_format` ("bhwc"/"hwct") plus an array's
        ndim to an explicit axes string, for the spatial family only.

        Parameters:
        -----------
        - ndim : int
            Number of dimensions of the input array (3, 4, or 5).
        - data_format : str
            "bhwc" or "hwct", required only when ndim == 4.

        Returns:
        --------
        - axes : str
            One of "hwc", "bhwc", "hwct", "bhwct".
    '''

    if ndim == 3:
        return "hwc"
    if ndim == 5:
        return "bhwct"
    if ndim == 4:
        fmt = data_format or "bhwc"
        if fmt not in ("bhwc", "hwct"):
            raise ValueError("data_format must be 'bhwc' or 'hwct' for 4D input")
        return fmt
    raise ValueError(f"Unsupported spatial ndim={ndim}")


def to_canonical(
    data: np.ndarray,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
) -> np.ndarray:
    '''
        Normalizes an input array to the canonical (batch, H, W, C, T) shape.

        Parameters:
        -----------
        - data : np.ndarray
            Input array, in one of two families: spatial (3D/4D/5D, see
            `data_format`) or series (1D/2D/3D with no spatial extent, see
            `axes`).
        - data_format : str
            For the spatial family only. Required only when data.ndim == 4,
            since (batch,H,W,C) and (H,W,C,T) are otherwise indistinguishable
            from shape alone. One of "bhwc" (default assumption) or "hwct".
        - axes : str
            For the series family only (pure time series with no spatial
            extent). Required whenever `data` represents one of the series
            shapes, since dimensionality alone cannot disambiguate them.
            One of "t" (T,), "c" (C,), "bt" (B,T), "ct" (C,T), "bct" (B,C,T).
            When given, `data_format` is ignored and `data` is expected to
            have exactly len(axes) dimensions.

        Returns:
        --------
        - canonical : np.ndarray
            Array reshaped to (batch, H, W, C, T), with any axis not
            present in the original input set to size 1 (including H/W
            for series inputs, which have no spatial extent at all).

        Usage:
        ------

        ```python
        import numpy as np
        from aule._shapes import to_canonical

        # spatial family, as before
        x = np.random.rand(10, 64, 64, 3)
        y = to_canonical(x)  # -> shape (10, 64, 64, 3, 1)

        # series family: a batch of multi-channel time series, no spatial extent
        s = np.random.rand(8, 4, 100)  # (B, C, T)
        s_c = to_canonical(s, axes="bct")  # -> shape (8, 1, 1, 4, 100)
        ```
    '''

    if axes is not None:
        return _series_to_canonical(data, axes)

    ndim = data.ndim

    if ndim == 1:
        raise ValueError(
            "1D input requires an explicit axes parameter ('t' for a time "
            "series, or 'c' for a single channel's worth of values), since "
            "a bare 1D array is otherwise ambiguous."
        )

    if ndim == 2:
        raise ValueError(
            "2D input requires an explicit axes parameter ('bt' for "
            "(batch, time) or 'ct' for (channel, time)), since a bare 2D "
            "array is otherwise ambiguous between the two."
        )

    if ndim == 3:
        # (H, W, C) -> (1, H, W, C, 1)
        return data[np.newaxis, :, :, :, np.newaxis]

    if ndim == 4:
        fmt = data_format or "bhwc"
        if fmt not in ("bhwc", "hwct"):
            raise ValueError("data_format must be 'bhwc' or 'hwct' for 4D input")
        if fmt == "bhwc":
            # (batch, H, W, C) -> (batch, H, W, C, 1)
            return data[:, :, :, :, np.newaxis]
        else:
            # (H, W, C, T) -> (1, H, W, C, T)
            return data[np.newaxis, :, :, :, :]

    if ndim == 5:
        # already (batch, H, W, C, T)
        return data

    raise ValueError(
        f"Unsupported array with ndim={ndim}. "
        "Expected 1D/2D series (with axes=...), 3D (H,W,C), "
        "4D (batch,H,W,C) or (H,W,C,T), or 5D (batch,H,W,C,T)."
    )


def _series_to_canonical(data: np.ndarray, axes: str) -> np.ndarray:
    '''
        Promotes a series-family array (no spatial extent) to the
        canonical (batch, H, W, C, T) shape with H=W=1, inserting size-1
        axes for batch/channel/time wherever they are absent from `axes`.

        Parameters:
        -----------
        - data : np.ndarray
            Array with exactly len(axes) dimensions.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" (order of axes must match
            the order of `data`'s dimensions).

        Returns:
        --------
        - canonical : np.ndarray
            Array reshaped to (batch, 1, 1, C, T).
    '''

    valid_axes_strings = ("t", "c", "bt", "ct", "bct")
    if axes not in valid_axes_strings:
        raise ValueError(
            f"Unknown axes '{axes}' for series input. Expected one of: "
            f"{', '.join(valid_axes_strings)}."
        )

    if data.ndim != len(axes):
        raise ValueError(
            f"axes='{axes}' implies {len(axes)} dimensions, but the input "
            f"array has ndim={data.ndim}."
        )

    logger.debug("Promoting series input with axes='%s' and shape %s to canonical form", axes, data.shape)

    dim_map = {letter: data.shape[i] for i, letter in enumerate(axes)}
    B = dim_map.get("b", 1)
    C = dim_map.get("c", 1)
    T = dim_map.get("t", 1)

    # Only 5 valid combinations exist, so handle each explicitly rather
    # than attempting a fully generic axis-insertion algorithm.
    if axes == "t":
        bct = data.reshape(1, 1, T)
    elif axes == "c":
        bct = data.reshape(1, C, 1)
    elif axes == "bt":
        bct = data.reshape(B, 1, T)
    elif axes == "ct":
        bct = data.reshape(C, T)[np.newaxis, :, :]
    elif axes == "bct":
        bct = data

    # (B, C, T) -> (B, 1, 1, C, T)
    return bct[:, np.newaxis, np.newaxis, :, :]


def is_degenerate_spatial(canonical: np.ndarray) -> bool:
    '''
        Checks whether a canonical (batch, H, W, C, T) array has no real
        spatial extent (H == W == 1), i.e. it originated from a series
        input rather than a genuine spatial field.

        Parameters:
        -----------
        - canonical : np.ndarray
            Array already normalized via `to_canonical`.

        Returns:
        --------
        - degenerate : bool
            True if H == W == 1.
    '''

    return canonical.shape[1] == 1 and canonical.shape[2] == 1


def is_degenerate_temporal(canonical: np.ndarray) -> bool:
    '''
        Checks whether a canonical (batch, H, W, C, T) array has no real
        temporal extent (T == 1).

        Parameters:
        -----------
        - canonical : np.ndarray
            Array already normalized via `to_canonical`.

        Returns:
        --------
        - degenerate : bool
            True if T == 1.
    '''

    return canonical.shape[4] == 1


def match_shapes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Normalizes a pair of arrays (ground truth and prediction) to the
        canonical (batch, H, W, C, T) shape and checks they are compatible.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, spatial or series family (see `to_canonical`).
        - y_pred : np.ndarray
            Prediction array, same family/shape as y_true.
        - data_format : str
            See `to_canonical`. Applied to both arrays (spatial family only).
        - axes : str
            See `to_canonical`. Applied to both arrays (series family only).

        Returns:
        --------
        - y_true_c, y_pred_c : tuple of np.ndarray
            Both arrays reshaped to the same canonical (batch, H, W, C, T) shape.

        Usage:
        ------

        ```python
        import numpy as np
        from aule._shapes import match_shapes

        gt   = np.random.rand(64, 64, 1)
        pred = np.random.rand(64, 64, 1)
        gt_c, pred_c = match_shapes(gt, pred)

        # series family
        gt_series   = np.random.rand(8, 4, 100)  # (B, C, T)
        pred_series = np.random.rand(8, 4, 100)
        gt_c, pred_c = match_shapes(gt_series, pred_series, axes="bct")
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format, axes=axes)
    y_pred_c = to_canonical(y_pred, data_format=data_format, axes=axes)

    if y_true_c.shape != y_pred_c.shape:
        raise ValueError(
            f"Shape mismatch after normalization: y_true {y_true_c.shape} "
            f"vs y_pred {y_pred_c.shape}"
        )

    return y_true_c, y_pred_c


def apply_nan_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ignore_nan: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Optionally replaces non-finite values (NaN, +-inf) with the
        per-array finite median, so downstream metrics are not corrupted
        by missing data.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array (any shape).
        - y_pred : np.ndarray
            Prediction array (same shape as y_true).
        - ignore_nan : bool
            If True, non-finite values in both arrays are replaced with
            their respective finite median (default: False, no handling).

        Returns:
        --------
        - y_true_clean, y_pred_clean : tuple of np.ndarray
            Arrays with non-finite values replaced if ignore_nan is True,
            otherwise the original arrays untouched.

        Usage:
        ------

        ```python
        import numpy as np
        from aule._shapes import apply_nan_mask

        gt   = np.array([1.0, np.nan, 3.0])
        pred = np.array([1.1, 2.1, np.nan])
        gt_clean, pred_clean = apply_nan_mask(gt, pred, ignore_nan=True)
        ```
    '''

    if not ignore_nan:
        return y_true, y_pred

    def _fill(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy().astype(np.float64)
        finite = arr[np.isfinite(arr)]
        fill_value = float(np.median(finite)) if finite.size > 0 else 0.0
        arr[~np.isfinite(arr)] = fill_value
        return arr

    return _fill(y_true), _fill(y_pred)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    '''
        Builds a boolean mask of positions that are finite in all given arrays.

        Parameters:
        -----------
        - *arrays : np.ndarray
            One or more arrays of the same shape.

        Returns:
        --------
        - mask : np.ndarray
            Boolean array, True where all inputs are finite.

        Usage:
        ------

        ```python
        import numpy as np
        from aule._shapes import finite_mask

        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, np.nan])
        mask = finite_mask(a, b)  # array([True, False, False])
        ```
    '''

    mask = np.isfinite(arrays[0])
    for arr in arrays[1:]:
        mask &= np.isfinite(arr)
    return mask
