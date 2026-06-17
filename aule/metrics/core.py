"""
    Core error metrics for comparing ground truth and predicted fields.

    All metrics accept numpy arrays in any of the 4 supported shapes:

        (a) (batch, H, W, C)
        (b) (batch, H, W, C, T)
        (c) (H, W, C)
        (d) (H, W, C, T)

    and reduce over all axes except channel, unless otherwise specified.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes, apply_nan_mask, finite_mask

__all__ = [
    "rmse", "mse", "mae", "bias", "pearson_r", "ssim", "psnr",
    "r2_score", "mape", "smape", "nse", "kge", "max_error", "explained_variance",
    "wasserstein_distance", "quantile_mapping_bias",
]


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Root Mean Squared Error between ground truth and prediction.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D (see _shapes.to_canonical).
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            RMSE value, lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import rmse

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        score = rmse(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    diff = y_true_c.astype(np.float64) - y_pred_c.astype(np.float64)

    if ignore_nan:
        mask = finite_mask(y_true_c, y_pred_c)
        diff = diff[mask]

    return float(np.sqrt(np.mean(diff ** 2)))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Mean Absolute Error between ground truth and prediction.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            MAE value, lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import mae

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        score = mae(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    diff = np.abs(y_true_c.astype(np.float64) - y_pred_c.astype(np.float64))

    if ignore_nan:
        mask = finite_mask(y_true_c, y_pred_c)
        diff = diff[mask]

    return float(np.mean(diff))


def bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the mean bias (prediction minus ground truth).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            Mean bias. Positive means the prediction overestimates the truth.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import bias

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + 0.05
        score = bias(gt, pred)  # ~ +0.05
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    diff = y_pred_c.astype(np.float64) - y_true_c.astype(np.float64)

    if ignore_nan:
        mask = finite_mask(y_true_c, y_pred_c)
        diff = diff[mask]

    return float(np.mean(diff))


def pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Pearson correlation coefficient between ground truth
        and prediction, flattened over all dimensions.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            Pearson r, in [-1, 1]. Higher is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import pearson_r

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        score = pearson_r(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]

    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    '''
        Builds a normalized 2D Gaussian window used by the SSIM computation.

        Parameters:
        -----------
        - window_size : int
            Side length of the (square) window (default: 11).
        - sigma : float
            Standard deviation of the Gaussian (default: 1.5).

        Returns:
        --------
        - window : np.ndarray
            2D array of shape (window_size, window_size), sums to 1.
    '''

    coords = np.arange(window_size) - window_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


def _ssim_2d(a: np.ndarray, b: np.ndarray, window: np.ndarray) -> float:
    '''
        Computes SSIM between two single-channel 2D fields using a fixed
        Gaussian window, implemented with a simple sliding-window approach
        (no external dependencies beyond numpy/scipy).

        Parameters:
        -----------
        - a : np.ndarray
            First 2D field (H, W).
        - b : np.ndarray
            Second 2D field (H, W), same shape as a.
        - window : np.ndarray
            2D Gaussian window, as returned by `_gaussian_window`.

        Returns:
        --------
        - value : float
            SSIM value in [-1, 1], higher is better.
    '''

    from scipy.signal import convolve2d

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    mu_a = convolve2d(a, window, mode="same", boundary="symm")
    mu_b = convolve2d(b, window, mode="same", boundary="symm")

    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = convolve2d(a * a, window, mode="same", boundary="symm") - mu_a2
    sigma_b2 = convolve2d(b * b, window, mode="same", boundary="symm") - mu_b2
    sigma_ab = convolve2d(a * b, window, mode="same", boundary="symm") - mu_ab

    sigma_a2 = np.clip(sigma_a2, a_min=0, a_max=None)
    sigma_b2 = np.clip(sigma_b2, a_min=0, a_max=None)

    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)

    return float(np.mean(numerator / np.clip(denominator, a_min=1e-12, a_max=None)))


def ssim(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    window_size: int = 11,
) -> float:
    '''
        Computes the Structural Similarity Index (SSIM) between ground truth
        and prediction, averaged over batch, channel and time.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are filled with the per-sample
            finite median before computing SSIM (default: False).
        - window_size : int
            Side length of the Gaussian window used for local statistics (default: 11).

        Returns:
        --------
        - value : float
            Mean SSIM over batch, channel and time. In [-1, 1], higher is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import ssim

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = ssim(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)
    y_true_c, y_pred_c = apply_nan_mask(y_true_c, y_pred_c, ignore_nan=ignore_nan)

    window = _gaussian_window(window_size)

    B, H, W, C, T = y_true_c.shape
    scores = []
    for b in range(B):
        for c in range(C):
            for t in range(T):
                a2d = y_true_c[b, :, :, c, t].astype(np.float64)
                b2d = y_pred_c[b, :, :, c, t].astype(np.float64)
                scores.append(_ssim_2d(a2d, b2d, window))

    return float(np.mean(scores))


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Mean Squared Error between ground truth and prediction.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            MSE value, lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import mse

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.1, gt.shape)
        score = mse(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    diff = y_true_c.astype(np.float64) - y_pred_c.astype(np.float64)

    if ignore_nan:
        mask = finite_mask(y_true_c, y_pred_c)
        diff = diff[mask]

    return float(np.mean(diff ** 2))


def psnr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_range: Optional[float] = None,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Peak Signal-to-Noise Ratio between ground truth and
        prediction, in decibels. Standard image-quality metric, also widely
        reported for super-resolution and downscaling tasks.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_range : float
            The value range of the data (max - min). If None (default),
            it is estimated from `y_true` as (max - min) over all finite values.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            PSNR in dB. Higher is better. Returns +inf when prediction is exact.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import psnr

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = psnr(gt, pred, data_range=1.0)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    diff = y_true_c.astype(np.float64) - y_pred_c.astype(np.float64)

    if ignore_nan:
        mask = finite_mask(y_true_c, y_pred_c)
        diff = diff[mask]
        true_finite = y_true_c.astype(np.float64)[mask]
    else:
        true_finite = y_true_c.astype(np.float64)

    mse_value = float(np.mean(diff ** 2))
    if mse_value == 0.0:
        return float("inf")

    if data_range is None:
        data_range = float(np.max(true_finite) - np.min(true_finite))
        if data_range == 0:
            data_range = 1.0

    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse_value))


def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the coefficient of determination (R^2): 1 minus the ratio
        of residual variance to total variance of the ground truth.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            R^2 score. 1.0 is a perfect fit, 0.0 means the model is as good
            as predicting the mean, negative values mean it is worse.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import r2_score

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = r2_score(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]

    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)

    if ss_tot == 0:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    eps: float = 1e-8,
) -> float:
    '''
        Computes the Mean Absolute Percentage Error. Sensitive to small
        ground truth values, since the error is divided by y_true; prefer
        `smape` when the data can be close to zero.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).
        - eps : float
            Small constant added to the denominator to avoid division by zero.

        Returns:
        --------
        - value : float
            MAPE, expressed as a percentage (e.g. 5.0 means 5%). Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import mape

        gt   = np.random.rand(8, 64, 64, 1) + 1.0
        pred = gt * 1.05
        score = mape(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64)
    b = y_pred_c.astype(np.float64)

    ratio = np.abs((a - b) / (np.abs(a) + eps))

    if ignore_nan:
        mask = finite_mask(a, b)
        ratio = ratio[mask]

    return float(np.mean(ratio) * 100.0)


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    eps: float = 1e-8,
) -> float:
    '''
        Computes the Symmetric Mean Absolute Percentage Error, bounded in
        [0, 200] and more stable than `mape` when values are close to zero.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).
        - eps : float
            Small constant added to the denominator to avoid division by zero.

        Returns:
        --------
        - value : float
            SMAPE, expressed as a percentage in [0, 200]. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import smape

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = smape(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64)
    b = y_pred_c.astype(np.float64)

    denom = (np.abs(a) + np.abs(b)) / 2.0 + eps
    ratio = np.abs(a - b) / denom

    if ignore_nan:
        mask = finite_mask(a, b)
        ratio = ratio[mask]

    return float(np.mean(ratio) * 100.0)


def nse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Nash-Sutcliffe Efficiency, a standard goodness-of-fit
        metric in hydrology and climate model validation, equivalent to
        `r2_score` but conventionally named differently in that literature.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            NSE. 1.0 is a perfect fit, 0.0 means as good as the observed mean,
            negative values mean worse than the mean.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import nse

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = nse(gt, pred)
        ```
    '''

    return r2_score(y_true, y_pred, data_format=data_format, ignore_nan=ignore_nan)


def kge(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> dict:
    '''
        Computes the Kling-Gupta Efficiency and its three components
        (correlation, variability ratio, bias ratio). Widely used in climate
        and hydrology model validation since it decomposes error into
        interpretable sources, unlike a single aggregate score.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - components : dict
            Dictionary with keys "kge" (overall score, 1.0 is perfect),
            "r" (Pearson correlation), "alpha" (std ratio, pred/true),
            "beta" (mean ratio, pred/true).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import kge

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        result = kge(gt, pred)
        print(result["kge"], result["r"], result["alpha"], result["beta"])
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]

    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a), np.std(b)

    r = float(np.corrcoef(a, b)[0, 1]) if (a.size >= 2 and std_a > 0 and std_b > 0) else float("nan")
    alpha = float(std_b / std_a) if std_a > 0 else float("nan")
    beta = float(mean_b / mean_a) if mean_a != 0 else float("nan")

    if any(np.isnan(v) for v in (r, alpha, beta)):
        score = float("nan")
    else:
        score = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    return {"kge": score, "r": r, "alpha": alpha, "beta": beta}


def max_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the maximum absolute error between ground truth and prediction.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            Maximum absolute error. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import max_error

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = max_error(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    diff = np.abs(y_true_c.astype(np.float64) - y_pred_c.astype(np.float64))

    if ignore_nan:
        mask = finite_mask(y_true_c, y_pred_c)
        diff = diff[mask]

    return float(np.max(diff))


def explained_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the fraction of ground truth variance explained by the
        prediction: 1 - Var(y_true - y_pred) / Var(y_true). Similar to
        `r2_score` but invariant to a constant additive bias.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            Explained variance score. 1.0 is a perfect fit.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import explained_variance

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = explained_variance(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]

    var_true = np.var(a)
    if var_true == 0:
        return float("nan")

    return float(1.0 - np.var(a - b) / var_true)


def wasserstein_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the 1D Wasserstein distance (Earth Mover's Distance)
        between the marginal distributions of ground truth and prediction,
        flattened over all dimensions. Unlike `ks_test`-style statistics,
        it accounts for the magnitude of distributional shifts, not just
        their location, making it a good fit for heavy-tailed data (e.g.
        precipitation, extreme events) where a few large discrepancies in
        the tail matter more than many small ones in the bulk.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            Wasserstein distance, in the same units as the data. Lower is better
            (0 means identical distributions).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import wasserstein_distance

        gt   = np.random.exponential(1.0, (8, 64, 64, 1))
        pred = np.random.exponential(1.1, (8, 64, 64, 1))
        score = wasserstein_distance(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)

    # interpolate onto a common quantile grid when sample sizes differ,
    # so the sorted-array difference below is well defined either way
    n = max(a_sorted.size, b_sorted.size)
    quantiles = np.linspace(0, 1, n)

    a_quantiles = np.quantile(a_sorted, quantiles)
    b_quantiles = np.quantile(b_sorted, quantiles)

    return float(np.mean(np.abs(a_quantiles - b_quantiles)))


def quantile_mapping_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_quantiles: int = 100,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the mean absolute difference between the quantile function
        (inverse CDF) of the prediction and that of the ground truth,
        evaluated at evenly spaced quantiles. This is the diagnostic
        typically used before applying quantile-mapping bias correction to
        climate model output: it tells you how much the prediction's full
        distribution (not just its mean) needs to be shifted/stretched to
        match observations.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - n_quantiles : int
            Number of evenly spaced quantiles to evaluate, in (1, 100) (default: 100).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the computation (default: False).

        Returns:
        --------
        - value : float
            Mean absolute quantile-function bias. Lower is better (0 means
            the two distributions match at every quantile evaluated).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import quantile_mapping_bias

        gt   = np.random.exponential(1.0, (8, 64, 64, 1))
        pred = gt * 1.1  # consistently overestimates magnitude
        score = quantile_mapping_bias(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = y_true_c.astype(np.float64).ravel()
    b = y_pred_c.astype(np.float64).ravel()

    if ignore_nan:
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

    quantile_levels = np.linspace(0, 1, n_quantiles)

    true_q = np.quantile(a, quantile_levels)
    pred_q = np.quantile(b, quantile_levels)

    return float(np.mean(np.abs(pred_q - true_q)))
