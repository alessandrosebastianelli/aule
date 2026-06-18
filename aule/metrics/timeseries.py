"""
    Metrics for pure time series data: (B,C,T), (B,T), (C,T), (T,), (C,).

    These functions accept the series-family shapes and use the `axes`
    parameter to disambiguate them. They are also callable on spatial data
    that has been canonicalized to (batch, H, W, C, T) with any H/W, since
    internally they reduce over the spatial axes before computing.

    All functions that iterate over channels or batch members use tqdm
    progress bars when the `tqdm` package is installed and the number of
    iterations exceeds a small threshold; otherwise they fall back silently
    to plain loops.
"""

from typing import Optional, Tuple
import numpy as np

from .._shapes import to_canonical, match_shapes
from .._logging import logger

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

_TQDM_THRESHOLD = 4   # show progress bar only when iterating over more than this many items

__all__ = [
    "lag_correlation",
    "cross_channel_correlation",
    "peak_timing_error",
    "dtw_distance",
]


def _progress(iterable, desc: str, total: int):
    """Wrap an iterable in tqdm when available and worthwhile."""
    if _HAS_TQDM and total > _TQDM_THRESHOLD:
        return _tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


def _spatial_mean(arr: np.ndarray, ignore_nan: bool) -> np.ndarray:
    """
    Reduce canonical (B, H, W, C, T) to (B, C, T) by averaging over H and W.
    """
    if ignore_nan:
        return np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=(1, 2))
    return np.mean(arr, axis=(1, 2))


def lag_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag: int = 20,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    ignore_nan: bool = False,
) -> np.ndarray:
    '''
        Computes the cross-correlation between ground truth and prediction
        at lags -max_lag … +max_lag, averaged over batch and channel.

        A positive lag L means the prediction leads the ground truth by L
        steps (prediction anticipates the signal); a negative lag means it
        lags behind. The lag at which the cross-correlation peaks tells you
        by how many time steps a model is ahead or behind the true signal.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth, any spatial or series shape.
        - y_pred : np.ndarray
            Prediction, same shape as y_true.
        - max_lag : int
            Maximum absolute lag to compute, in time steps (default: 20).
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs (see `to_canonical`).
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - ignore_nan : bool
            If True, NaN values are excluded before computing correlations (default: False).

        Returns:
        --------
        - corr : np.ndarray
            1D array of length 2*max_lag+1 with cross-correlation values at
            lags [-max_lag, ..., 0, ..., +max_lag].

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import lag_correlation

        # pure time series (B, C, T)
        gt   = np.random.randn(4, 2, 200)
        pred = np.roll(gt, shift=3, axis=-1) + 0.1 * np.random.randn(*gt.shape)
        corr = lag_correlation(gt, pred, max_lag=30, axes="bct")
        # peak should be near lag=+3 (prediction leads by 3 steps)

        # or spatial data
        gt2 = np.random.randn(4, 32, 32, 1, 50)
        pred2 = np.roll(gt2, shift=2, axis=-1)
        corr2 = lag_correlation(gt2, pred2, max_lag=10)
        ```
    '''
    logger.debug("lag_correlation: max_lag=%d, axes=%s", max_lag, axes)

    y_true_c = to_canonical(y_true, data_format=data_format, axes=axes)
    y_pred_c = to_canonical(y_pred, data_format=data_format, axes=axes)

    T = y_true_c.shape[-1]
    if T < 2:
        raise ValueError(
            f"lag_correlation requires at least 2 time steps, got T={T}. "
            "Pass axes='bct' (or similar) if your input is a series."
        )
    if max_lag >= T:
        raise ValueError(f"max_lag={max_lag} must be less than T={T}.")

    # (B, H, W, C, T) -> (B, C, T)
    a = _spatial_mean(y_true_c.astype(np.float64), ignore_nan)
    b = _spatial_mean(y_pred_c.astype(np.float64), ignore_nan)

    B, C, _ = a.shape
    lags = np.arange(-max_lag, max_lag + 1)
    corr_sum = np.zeros(len(lags), dtype=np.float64)
    n_pairs = 0

    for bi in _progress(range(B), "lag_correlation [batch]", B):
        for ci in range(C):
            ta = a[bi, ci]
            tb = b[bi, ci]
            if ignore_nan:
                valid = np.isfinite(ta) & np.isfinite(tb)
                ta, tb = ta[valid], tb[valid]
            if ta.size < 2:
                continue
            ta = ta - np.mean(ta)
            tb = tb - np.mean(tb)
            denom = np.sqrt(np.dot(ta, ta) * np.dot(tb, tb))
            if denom == 0:
                continue
            for k, lag in enumerate(lags):
                if lag == 0:
                    corr_sum[k] += np.dot(ta, tb) / denom
                elif lag > 0:
                    if lag < len(ta):
                        corr_sum[k] += np.dot(ta[:-lag], tb[lag:]) / denom
                else:  # lag < 0
                    l = -lag
                    if l < len(ta):
                        corr_sum[k] += np.dot(ta[l:], tb[:-l]) / denom
            n_pairs += 1

    if n_pairs == 0:
        return np.full(len(lags), np.nan)

    return corr_sum / n_pairs


def cross_channel_correlation(
    data: np.ndarray,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    ignore_nan: bool = False,
) -> np.ndarray:
    '''
        Computes the Pearson correlation matrix between channels (C×C),
        treating time/batch as the sample dimension.  Useful for comparing
        multi-variable series (e.g. different meteorological stations or
        sensor bands) to see whether a model reproduces the inter-variable
        structure.

        Parameters:
        -----------
        - data : np.ndarray
            Input array, any spatial or series shape.
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - ignore_nan : bool
            If True, NaN values are excluded (default: False).

        Returns:
        --------
        - corr_matrix : np.ndarray
            Square matrix of shape (C, C) with Pearson correlation values.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import cross_channel_correlation

        data = np.random.randn(4, 5, 100)  # (B, C=5, T=100)
        corr = cross_channel_correlation(data, axes="bct")  # (5, 5)
        ```
    '''
    logger.debug("cross_channel_correlation: axes=%s, shape=%s", axes, data.shape)

    data_c = to_canonical(data, data_format=data_format, axes=axes)
    B, H, W, C, T = data_c.shape

    # (B, H, W, C, T) -> (C, B*H*W*T) to correlate channels
    flat = data_c.astype(np.float64).transpose(3, 0, 1, 2, 4).reshape(C, -1)

    if ignore_nan:
        corr = np.full((C, C), np.nan)
        for i in _progress(range(C), "cross_channel_correlation", C):
            for j in range(i, C):
                a, b = flat[i], flat[j]
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 2:
                    continue
                r = float(np.corrcoef(a[mask], b[mask])[0, 1]) if C > 1 else 1.0
                corr[i, j] = corr[j, i] = r
        np.fill_diagonal(corr, 1.0)
    else:
        if C == 1:
            corr = np.array([[1.0]])
        else:
            corr = np.corrcoef(flat)

    return corr


def peak_timing_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the mean absolute difference in peak location (time step
        with the maximum value) between ground truth and prediction, averaged
        over batch and channel.  Useful for evaluating event-timing accuracy:
        does the model predict floods, heatwaves, or emission peaks at the
        right time, even if the amplitude is close?

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth, any spatial or series shape.
        - y_pred : np.ndarray
            Prediction, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - ignore_nan : bool
            If True, NaN values are excluded before argmax (default: False).

        Returns:
        --------
        - value : float
            Mean absolute difference in peak time step. Lower is better
            (0 means the model always peaks at the exact right time step).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import peak_timing_error

        t = np.linspace(0, 4 * np.pi, 200)
        gt   = np.sin(t).reshape(1, 1, 200)   # (B=1, C=1, T=200)
        pred = np.sin(t - 0.3).reshape(1, 1, 200)  # shifted by ~10 steps
        err  = peak_timing_error(gt, pred, axes="bct")  # ~10
        ```
    '''
    logger.debug("peak_timing_error: axes=%s", axes)

    y_true_c = to_canonical(y_true, data_format=data_format, axes=axes)
    y_pred_c = to_canonical(y_pred, data_format=data_format, axes=axes)

    T = y_true_c.shape[-1]
    if T < 2:
        raise ValueError(f"peak_timing_error requires T≥2, got T={T}.")

    a = _spatial_mean(y_true_c.astype(np.float64), ignore_nan)
    b = _spatial_mean(y_pred_c.astype(np.float64), ignore_nan)

    B, C, _ = a.shape
    errors = []

    for bi in range(B):
        for ci in range(C):
            ta = a[bi, ci]
            tb = b[bi, ci]
            if ignore_nan:
                valid_a = np.isfinite(ta)
                valid_b = np.isfinite(tb)
                if valid_a.sum() == 0 or valid_b.sum() == 0:
                    continue
                # replace NaN with -inf so argmax ignores them
                ta = np.where(valid_a, ta, -np.inf)
                tb = np.where(valid_b, tb, -np.inf)
            errors.append(abs(int(np.argmax(ta)) - int(np.argmax(tb))))

    if not errors:
        return float("nan")
    return float(np.mean(errors))


def dtw_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    axes: Optional[str] = None,
    ignore_nan: bool = False,
    window: Optional[int] = None,
) -> float:
    '''
        Computes the Dynamic Time Warping (DTW) distance between ground
        truth and prediction, averaged over batch and channel.  DTW finds
        the minimum-cost alignment between two time series by allowing
        non-linear warping of the time axis, making it much more tolerant
        of temporal shifts and distortions than direct pointwise error
        metrics like RMSE.  It is the right choice when you care about
        shape similarity rather than exact time registration.

        Uses the O(N*M) standard DTW algorithm with an optional Sakoe-Chiba
        band constraint (`window`) to limit the warping path to a diagonal
        corridor, reducing both computation cost and overly flexible
        alignments for long series.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth, any spatial or series shape.
        - y_pred : np.ndarray
            Prediction, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct" for 4D spatial inputs.
        - axes : str
            One of "t", "c", "bt", "ct", "bct" for series inputs.
        - ignore_nan : bool
            If True, NaN values are replaced with linear interpolation
            before computing DTW (default: False).
        - window : int
            Sakoe-Chiba band width (max allowed time-warp steps). If None
            (default), no band constraint is applied (full DTW).

        Returns:
        --------
        - value : float
            Mean DTW distance over batch/channel. Lower is better (0 means
            identical time series up to allowed warping).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import dtw_distance

        t = np.linspace(0, 4*np.pi, 100)
        gt   = np.sin(t).reshape(1, 1, 100)
        pred = np.sin(t * 1.1).reshape(1, 1, 100)   # slightly different frequency
        dist = dtw_distance(gt, pred, axes="bct")
        ```
    '''
    logger.debug("dtw_distance: axes=%s, window=%s", axes, window)

    y_true_c = to_canonical(y_true, data_format=data_format, axes=axes)
    y_pred_c = to_canonical(y_pred, data_format=data_format, axes=axes)

    T = y_true_c.shape[-1]
    if T < 2:
        raise ValueError(f"dtw_distance requires T≥2, got T={T}.")

    a = _spatial_mean(y_true_c.astype(np.float64), ignore_nan)
    b = _spatial_mean(y_pred_c.astype(np.float64), ignore_nan)

    B, C, _ = a.shape
    distances = []

    for bi in _progress(range(B), "dtw_distance [batch]", B):
        for ci in range(C):
            ta = a[bi, ci]
            tb = b[bi, ci]

            if ignore_nan:
                ta = _interpolate_nan(ta)
                tb = _interpolate_nan(tb)

            distances.append(_dtw_1d(ta, tb, window=window))

    if not distances:
        return float("nan")
    return float(np.mean(distances))


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    """Linear interpolation to fill NaN values in a 1D array."""
    x = x.copy()
    nans = ~np.isfinite(x)
    if not np.any(nans):
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def _dtw_1d(a: np.ndarray, b: np.ndarray, window: Optional[int]) -> float:
    """Standard O(N*M) DTW with optional Sakoe-Chiba band."""
    N, M = len(a), len(b)
    w = max(window, abs(N - M)) if window is not None else max(N, M)

    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, N + 1):
        j_start = max(1, i - w)
        j_end = min(M, i + w)
        for j in range(j_start, j_end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[N, M])
