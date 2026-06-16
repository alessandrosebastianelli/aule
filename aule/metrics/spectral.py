"""
    Spectral and spatial-gradient error metrics.

    These metrics are particularly relevant for earth observation and climate
    model validation, where periodic artifacts (e.g. from upscaling) and edge
    sharpness (e.g. coastlines, orographic fronts) matter beyond plain pixel error.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes, apply_nan_mask

__all__ = ["spectral_error", "gradient_error"]


def spectral_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the L1 error between the 2D FFT power spectra of ground
        truth and prediction, averaged over batch, channel and time.
        Useful for detecting periodic artifacts (e.g. checkerboard patterns
        from upsampling) that pixel-wise metrics may miss.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are replaced with the finite median
            before computing the FFT (default: False).

        Returns:
        --------
        - value : float
            Mean L1 distance between normalized power spectra. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import spectral_error

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = spectral_error(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)
    y_true_c, y_pred_c = apply_nan_mask(y_true_c, y_pred_c, ignore_nan=ignore_nan)

    H, W = y_true_c.shape[1], y_true_c.shape[2]

    true_fft = np.fft.rfft2(y_true_c.astype(np.float64), axes=(1, 2))
    pred_fft = np.fft.rfft2(y_pred_c.astype(np.float64), axes=(1, 2))

    true_psd = np.abs(true_fft) / (H * W)
    pred_psd = np.abs(pred_fft) / (H * W)

    return float(np.mean(np.abs(true_psd - pred_psd)))


def _sobel_kernels() -> tuple:
    '''
        Builds the horizontal and vertical Sobel kernels used for gradient
        computation.

        Returns:
        --------
        - (kx, ky) : tuple of np.ndarray
            Two (3, 3) Sobel kernels, for the x and y directions.
    '''

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    return kx, ky


def _sobel_gradient(field: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> tuple:
    '''
        Computes horizontal and vertical Sobel gradients of a 2D field.

        Parameters:
        -----------
        - field : np.ndarray
            2D array (H, W).
        - kx, ky : np.ndarray
            Sobel kernels, as returned by `_sobel_kernels`.

        Returns:
        --------
        - (gx, gy) : tuple of np.ndarray
            Gradient fields in the x and y directions, same shape as input.
    '''

    from scipy.signal import convolve2d

    gx = convolve2d(field, kx, mode="same", boundary="symm")
    gy = convolve2d(field, ky, mode="same", boundary="symm")
    return gx, gy


def gradient_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    norm: str = "l1",
) -> float:
    '''
        Computes the Sobel gradient error between ground truth and prediction,
        averaged over batch, channel and time. Penalizes errors on spatial
        gradients (edges, coastlines, orographic fronts) rather than raw values.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are replaced with the finite median
            before computing gradients (default: False).
        - norm : str
            Either "l1" or "l2" (default: "l1").

        Returns:
        --------
        - value : float
            Mean gradient error. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import gradient_error

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = gradient_error(gt, pred, norm="l1")
        ```
    '''

    if norm not in ("l1", "l2"):
        raise ValueError("norm must be 'l1' or 'l2'")

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)
    y_true_c, y_pred_c = apply_nan_mask(y_true_c, y_pred_c, ignore_nan=ignore_nan)

    kx, ky = _sobel_kernels()

    B, H, W, C, T = y_true_c.shape
    errors = []
    for b in range(B):
        for c in range(C):
            for t in range(T):
                true2d = y_true_c[b, :, :, c, t].astype(np.float64)
                pred2d = y_pred_c[b, :, :, c, t].astype(np.float64)

                tgx, tgy = _sobel_gradient(true2d, kx, ky)
                pgx, pgy = _sobel_gradient(pred2d, kx, ky)

                if norm == "l1":
                    err = (np.mean(np.abs(pgx - tgx)) + np.mean(np.abs(pgy - tgy))) / 2.0
                else:
                    err = (np.mean((pgx - tgx) ** 2) + np.mean((pgy - tgy) ** 2)) / 2.0

                errors.append(err)

    return float(np.mean(errors))
