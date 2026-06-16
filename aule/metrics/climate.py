"""
    Climate-science specific metrics, operating along the time axis (T).

    These metrics require the temporal dimension to be meaningful (i.e. input
    shapes (batch, H, W, C, T) or (H, W, C, T)); for inputs without a time
    axis, T defaults to 1 and these metrics degrade gracefully but are less
    informative.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes, apply_nan_mask, finite_mask

__all__ = ["seasonal_error", "percentile_error", "pixelwise_temporal_correlation"]


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
