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


def to_canonical(data: np.ndarray, data_format: Optional[str] = None) -> np.ndarray:
    '''
        Normalizes an input array to the canonical (batch, H, W, C, T) shape.

        Parameters:
        -----------
        - data : np.ndarray
            Input array with 3, 4 or 5 dimensions, matching one of the
            supported shapes (H,W,C), (H,W,C,T), (batch,H,W,C), (batch,H,W,C,T)
        - data_format : str
            Required only when data.ndim == 4, since (batch,H,W,C) and
            (H,W,C,T) are otherwise indistinguishable from shape alone.
            One of "bhwc" (default assumption) or "hwct".

        Returns:
        --------
        - canonical : np.ndarray
            Array reshaped to (batch, H, W, C, T), with batch and/or T set
            to 1 when not present in the original input.

        Usage:
        ------

        ```python
        import numpy as np
        from aule._shapes import to_canonical

        x = np.random.rand(10, 64, 64, 3)
        y = to_canonical(x)  # -> shape (10, 64, 64, 3, 1), data_format defaults to "bhwc"
        ```
    '''

    if data.ndim == 3:
        # (H, W, C) -> (1, H, W, C, 1)
        return data[np.newaxis, :, :, :, np.newaxis]

    if data.ndim == 4:
        fmt = data_format or "bhwc"
        if fmt not in ("bhwc", "hwct"):
            raise ValueError("data_format must be 'bhwc' or 'hwct' for 4D input")
        if fmt == "bhwc":
            # (batch, H, W, C) -> (batch, H, W, C, 1)
            return data[:, :, :, :, np.newaxis]
        else:
            # (H, W, C, T) -> (1, H, W, C, T)
            return data[np.newaxis, :, :, :, :]

    if data.ndim == 5:
        # already (batch, H, W, C, T)
        return data

    raise ValueError(
        f"Unsupported array with ndim={data.ndim}. "
        "Expected 3D (H,W,C), 4D (batch,H,W,C) or (H,W,C,T), or 5D (batch,H,W,C,T)."
    )


def match_shapes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Normalizes a pair of arrays (ground truth and prediction) to the
        canonical (batch, H, W, C, T) shape and checks they are compatible.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, any of the 4 supported shapes.
        - data_format : str
            See `to_canonical`. Applied to both arrays.

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
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

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
