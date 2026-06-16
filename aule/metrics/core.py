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

__all__ = ["rmse", "mae", "bias", "pearson_r", "ssim"]


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
