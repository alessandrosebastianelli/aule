"""
    Climate-science specific metrics, operating along the time axis (T).

    These metrics require the temporal dimension to be meaningful (i.e. input
    shapes (batch, H, W, C, T) or (H, W, C, T)); for inputs without a time
    axis, T defaults to 1 and these metrics degrade gracefully but are less
    informative.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes, to_canonical, apply_nan_mask, finite_mask

__all__ = [
    "seasonal_error", "percentile_error", "pixelwise_temporal_correlation",
    "trend_error", "extreme_event_duration_error", "autocorrelation_error",
    "wet_day_frequency_error", "dry_spell_error", "anomaly_correlation_coefficient",
]


def seasonal_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the MSE between the spatial-mean time series of ground
        truth and prediction (i.e. how well the model preserves the
        temporal/seasonal trend, regardless of spatial detail).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded when computing spatial
            means (default: False).

        Returns:
        --------
        - value : float
            MSE between spatial-mean time series. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import seasonal_error

        gt   = np.random.rand(64, 64, 1, 365)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = seasonal_error(gt, pred, data_format="hwct")
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    if ignore_nan:
        true_mean = np.nanmean(np.where(np.isfinite(y_true_c), y_true_c, np.nan), axis=(1, 2))
        pred_mean = np.nanmean(np.where(np.isfinite(y_pred_c), y_pred_c, np.nan), axis=(1, 2))
    else:
        true_mean = np.mean(y_true_c, axis=(1, 2))
        pred_mean = np.mean(y_pred_c, axis=(1, 2))

    return float(np.mean((true_mean - pred_mean) ** 2))


def percentile_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float = 95.0,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the absolute error between a given percentile of the ground
        truth distribution and the same percentile of the prediction
        distribution. Useful for assessing how well a model captures
        extreme values (e.g. P95, P99 for heatwaves; P5, P1 for cold extremes).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - percentile : float
            Percentile to compare, in [0, 100] (default: 95.0).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the percentile
            computation (default: False).

        Returns:
        --------
        - value : float
            Absolute difference between the two percentile values. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import percentile_error

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt * 0.9
        score = percentile_error(gt, pred, percentile=95.0)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    true_flat = y_true_c.astype(np.float64).ravel()
    pred_flat = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        true_flat = true_flat[np.isfinite(true_flat)]
        pred_flat = pred_flat[np.isfinite(pred_flat)]

    true_p = np.percentile(true_flat, percentile)
    pred_p = np.percentile(pred_flat, percentile)

    return float(np.abs(true_p - pred_p))


def pixelwise_temporal_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> np.ndarray:
    '''
        Computes the Pearson correlation coefficient at each pixel, across
        the batch/time dimension, between ground truth and prediction.
        Requires a meaningful sample dimension (batch or time) to correlate over.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the per-pixel
            statistics (default: False).

        Returns:
        --------
        - r_map : np.ndarray
            Array of shape (H, W, C) with the Pearson r at each pixel,
            averaged over the time axis if both batch and time are present.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import pixelwise_temporal_correlation

        gt   = np.random.rand(100, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        r_map = pixelwise_temporal_correlation(gt, pred)  # shape (64, 64, 1)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)
    B, H, W, C, T = y_true_c.shape

    # Merge batch and time into a single "sample" axis (N, H, W, C) to correlate over.
    # Move the (B, T) axes to the front, then flatten them together.
    true_samples = np.moveaxis(y_true_c, 4, 1).reshape(B * T, H, W, C)
    pred_samples = np.moveaxis(y_pred_c, 4, 1).reshape(B * T, H, W, C)

    n = true_samples.shape[0]
    if n < 2:
        raise ValueError(
            "pixelwise_temporal_correlation requires at least 2 samples "
            "along batch or time; got a single sample."
        )

    if ignore_nan:
        valid = finite_mask(true_samples, pred_samples)
        true_samples = np.where(valid, true_samples, np.nan)
        pred_samples = np.where(valid, pred_samples, np.nan)
        mean_fn, sum_fn = np.nanmean, np.nansum
    else:
        mean_fn, sum_fn = np.mean, np.sum

    true_mean = mean_fn(true_samples, axis=0)
    pred_mean = mean_fn(pred_samples, axis=0)

    true_centered = true_samples - true_mean
    pred_centered = pred_samples - pred_mean

    numerator = sum_fn(true_centered * pred_centered, axis=0)
    true_var = sum_fn(true_centered ** 2, axis=0)
    pred_var = sum_fn(pred_centered ** 2, axis=0)
    denominator = np.sqrt(np.clip(true_var * pred_var, a_min=1e-12, a_max=None))

    r_map = np.clip(numerator / denominator, -1.0, 1.0)

    return r_map


def trend_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the absolute difference between the linear trend (slope of
        a least-squares fit over time) of the ground truth spatial-mean time
        series and that of the prediction. Useful for checking whether a
        model reproduces long-term warming/cooling or drying/wetting trends,
        independent of how well it matches short-term variability.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, with a time axis (shapes (b) or (d)).
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Must be
            "hwct" here, since a time axis is required.
        - ignore_nan : bool
            If True, non-finite values are excluded from the spatial means (default: False).

        Returns:
        --------
        - value : float
            Absolute difference between the two trend slopes (units per time step).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import trend_error

        t = np.arange(100)
        gt   = np.tile((0.01 * t).reshape(1, 1, 1, -1), (32, 32, 1, 1))
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = trend_error(gt, pred, data_format="hwct")
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    if y_true_c.shape[-1] < 2:
        raise ValueError(
            "trend_error requires at least 2 time steps; "
            f"got T={y_true_c.shape[-1]}. Did you forget data_format='hwct'?"
        )

    if ignore_nan:
        true_mean = np.nanmean(np.where(np.isfinite(y_true_c), y_true_c, np.nan), axis=(0, 1, 2, 3))
        pred_mean = np.nanmean(np.where(np.isfinite(y_pred_c), y_pred_c, np.nan), axis=(0, 1, 2, 3))
    else:
        true_mean = np.mean(y_true_c, axis=(0, 1, 2, 3))
        pred_mean = np.mean(y_pred_c, axis=(0, 1, 2, 3))

    t = np.arange(len(true_mean), dtype=np.float64)

    true_slope = float(np.polyfit(t, true_mean, 1)[0])
    pred_slope = float(np.polyfit(t, pred_mean, 1)[0])

    return float(np.abs(true_slope - pred_slope))


def _event_durations(series: np.ndarray, threshold: float, above: bool) -> np.ndarray:
    '''
        Finds the durations (in time steps) of consecutive runs where a
        1D series is above (or below) a threshold.

        Parameters:
        -----------
        - series : np.ndarray
            1D time series.
        - threshold : float
            Threshold value.
        - above : bool
            If True, finds runs where series > threshold; otherwise series < threshold.

        Returns:
        --------
        - durations : np.ndarray
            1D array of run lengths (in time steps). Empty if no event occurs.
    '''

    mask = series > threshold if above else series < threshold
    if not np.any(mask):
        return np.array([], dtype=np.int64)

    padded = np.concatenate(([0], mask.astype(np.int64), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    return (ends - starts).astype(np.int64)


def extreme_event_duration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    above: bool = True,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the absolute difference in mean extreme-event duration
        (e.g. heatwave or cold-wave length) between ground truth and
        prediction, based on the spatial-mean time series crossing a
        fixed threshold. An "event" is a run of consecutive time steps
        above (or below) the threshold.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, with a time axis (shapes (b) or (d)).
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - threshold : float
            Threshold value defining an extreme event (e.g. a heatwave temperature).
        - above : bool
            If True (default), an event is a run above the threshold
            (e.g. heatwave); if False, a run below it (e.g. cold wave).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Must be
            "hwct" here, since a time axis is required.
        - ignore_nan : bool
            If True, non-finite values are excluded from the spatial means (default: False).

        Returns:
        --------
        - value : float
            Absolute difference in mean event duration (in time steps).
            Returns 0.0 if neither series has any qualifying event.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import extreme_event_duration_error

        gt   = np.random.rand(32, 32, 1, 60) + 0.5
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = extreme_event_duration_error(gt, pred, threshold=0.8, data_format="hwct")
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    if ignore_nan:
        true_mean = np.nanmean(np.where(np.isfinite(y_true_c), y_true_c, np.nan), axis=(0, 1, 2, 3))
        pred_mean = np.nanmean(np.where(np.isfinite(y_pred_c), y_pred_c, np.nan), axis=(0, 1, 2, 3))
    else:
        true_mean = np.mean(y_true_c, axis=(0, 1, 2, 3))
        pred_mean = np.mean(y_pred_c, axis=(0, 1, 2, 3))

    true_durations = _event_durations(true_mean, threshold, above)
    pred_durations = _event_durations(pred_mean, threshold, above)

    true_avg = float(np.mean(true_durations)) if true_durations.size > 0 else 0.0
    pred_avg = float(np.mean(pred_durations)) if pred_durations.size > 0 else 0.0

    return float(np.abs(true_avg - pred_avg))


def autocorrelation_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag: int = 10,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the mean absolute error between the temporal autocorrelation
        function (ACF) of the ground truth spatial-mean time series and that
        of the prediction, up to a maximum lag. Useful for checking whether
        a model preserves temporal persistence/memory (e.g. drought
        persistence, slow climate modes).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, with a time axis (shapes (b) or (d)).
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - max_lag : int
            Maximum lag (in time steps) to compare (default: 10).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Must be
            "hwct" here, since a time axis is required.
        - ignore_nan : bool
            If True, non-finite values are excluded from the spatial means (default: False).

        Returns:
        --------
        - value : float
            Mean absolute error between the two ACFs over lags 1..max_lag. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import autocorrelation_error

        gt   = np.cumsum(np.random.randn(32, 32, 1, 200), axis=-1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        score = autocorrelation_error(gt, pred, max_lag=10, data_format="hwct")
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    T = y_true_c.shape[-1]
    if T <= max_lag:
        raise ValueError(
            f"autocorrelation_error requires more time steps than max_lag; "
            f"got T={T}, max_lag={max_lag}."
        )

    if ignore_nan:
        true_mean = np.nanmean(np.where(np.isfinite(y_true_c), y_true_c, np.nan), axis=(0, 1, 2, 3))
        pred_mean = np.nanmean(np.where(np.isfinite(y_pred_c), y_pred_c, np.nan), axis=(0, 1, 2, 3))
    else:
        true_mean = np.mean(y_true_c, axis=(0, 1, 2, 3))
        pred_mean = np.mean(y_pred_c, axis=(0, 1, 2, 3))

    def _acf(series: np.ndarray, max_lag: int) -> np.ndarray:
        series = series - np.mean(series)
        var = np.dot(series, series)
        if var == 0:
            return np.zeros(max_lag)
        return np.array([np.dot(series[:-lag], series[lag:]) / var for lag in range(1, max_lag + 1)])

    true_acf = _acf(true_mean, max_lag)
    pred_acf = _acf(pred_mean, max_lag)

    return float(np.mean(np.abs(true_acf - pred_acf)))


def wet_day_frequency_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1.0,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the absolute difference in "wet day" frequency (the
        fraction of time steps where the spatial-mean value exceeds a
        threshold, e.g. a precipitation amount) between ground truth and
        prediction. A standard precipitation-model diagnostic, since models
        often get total precipitation right while drastically over- or
        under-estimating how many days actually have rain ("drizzle bias").

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, with a time axis (shapes (b) or (d)).
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - threshold : float
            Value above which a time step counts as a "wet day" (default: 1.0).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Must be
            "hwct" here, since a time axis is required.
        - ignore_nan : bool
            If True, non-finite values are excluded from the spatial means (default: False).

        Returns:
        --------
        - value : float
            Absolute difference in wet-day frequency, in [0, 1]. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import wet_day_frequency_error

        gt   = np.random.exponential(1.0, (32, 32, 1, 60))
        pred = gt * 0.9
        score = wet_day_frequency_error(gt, pred, threshold=1.0, data_format="hwct")
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    if ignore_nan:
        true_mean = np.nanmean(np.where(np.isfinite(y_true_c), y_true_c, np.nan), axis=(0, 1, 2, 3))
        pred_mean = np.nanmean(np.where(np.isfinite(y_pred_c), y_pred_c, np.nan), axis=(0, 1, 2, 3))
    else:
        true_mean = np.mean(y_true_c, axis=(0, 1, 2, 3))
        pred_mean = np.mean(y_pred_c, axis=(0, 1, 2, 3))

    true_freq = float(np.mean(true_mean > threshold))
    pred_freq = float(np.mean(pred_mean > threshold))

    return float(np.abs(true_freq - pred_freq))


def dry_spell_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 1.0,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the absolute difference in mean dry-spell length (runs of
        consecutive time steps with the spatial-mean value below a
        threshold, e.g. days without meaningful precipitation) between
        ground truth and prediction. Complements `extreme_event_duration_error`,
        which is threshold-direction-agnostic but requires the caller to set
        `above=False`; this is a convenience wrapper specialized for the
        common dry-spell use case with a sensible default threshold.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, with a time axis (shapes (b) or (d)).
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - threshold : float
            Value below which a time step counts as "dry" (default: 1.0).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Must be
            "hwct" here, since a time axis is required.
        - ignore_nan : bool
            If True, non-finite values are excluded from the spatial means (default: False).

        Returns:
        --------
        - value : float
            Absolute difference in mean dry-spell duration (in time steps).
            Returns 0.0 if neither series has any qualifying dry spell.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import dry_spell_error

        gt   = np.random.exponential(1.0, (32, 32, 1, 60))
        pred = gt * 1.1
        score = dry_spell_error(gt, pred, threshold=1.0, data_format="hwct")
        ```
    '''

    return extreme_event_duration_error(
        y_true, y_pred, threshold=threshold, above=False,
        data_format=data_format, ignore_nan=ignore_nan,
    )


def anomaly_correlation_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    climatology: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Anomaly Correlation Coefficient (ACC): the Pearson
        correlation between ground truth and prediction *anomalies*
        (deviations from a reference climatology), rather than between the
        raw values. Standard verification metric in numerical weather
        prediction and seasonal/climate forecasting, since it isolates a
        model's skill at predicting departures from normal rather than
        simply tracking the climatological seasonal cycle (which any
        trivial model could get right).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - climatology : np.ndarray
            Reference climatology, broadcastable against y_true/y_pred
            after canonicalization (e.g. a per-pixel long-term mean with
            the same H/W/C as y_true, possibly with batch/time size 1).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D. Applied to
            y_true, y_pred and climatology.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            ACC in [-1, 1]. Higher is better; values near 1 indicate the
            model correctly predicts the sign and magnitude of anomalies.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import anomaly_correlation_coefficient

        climatology = np.random.rand(1, 32, 32, 1, 1)  # long-term mean, broadcastable
        gt   = climatology + np.random.normal(0, 0.2, (10, 32, 32, 1, 1))
        pred = climatology + np.random.normal(0, 0.2, (10, 32, 32, 1, 1))
        score = anomaly_correlation_coefficient(gt, pred, climatology)
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)
    clim_c = to_canonical(climatology, data_format=data_format)

    true_anomaly = y_true_c.astype(np.float64) - clim_c.astype(np.float64)
    pred_anomaly = y_pred_c.astype(np.float64) - clim_c.astype(np.float64)

    true_anomaly, pred_anomaly = np.broadcast_arrays(true_anomaly, pred_anomaly)

    a = true_anomaly.ravel()
    b = pred_anomaly.ravel()

    if ignore_nan:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]

    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])
