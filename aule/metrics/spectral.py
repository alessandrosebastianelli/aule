"""
    Spectral and spatial-gradient error metrics.

    These metrics are particularly relevant for earth observation and climate
    model validation, where periodic artifacts (e.g. from upscaling) and edge
    sharpness (e.g. coastlines, orographic fronts) matter beyond plain pixel error.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes, apply_nan_mask
from .._guards import requires
from .._logging import logger

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

_TQDM_THRESHOLD = 4


def _progress(iterable, desc: str, total: int):
    if _HAS_TQDM and total > _TQDM_THRESHOLD:
        return _tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable

__all__ = ["spectral_error", "gradient_error", "psd_radial_error", "spectral_angle_mapper"]


@requires(spatial=True)
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


@requires(spatial=True)
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
    logger.debug("gradient_error: %d slices (B=%d, C=%d, T=%d)", B*C*T, B, C, T)
    errors = []
    for b in _progress(range(B), "gradient_error [batch]", B):
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


def _radial_average(psd_2d: np.ndarray) -> np.ndarray:
    '''
        Computes the radially-averaged profile of a 2D power spectrum,
        binning frequencies by their integer distance from the zero-frequency
        corner.

        Parameters:
        -----------
        - psd_2d : np.ndarray
            2D power spectrum (H, W_half), as returned by rfft2 magnitude.

        Returns:
        --------
        - profile : np.ndarray
            1D array, the mean power at each radial frequency bin.
    '''

    H, W = psd_2d.shape
    y, x = np.indices((H, W))
    # distance from the (0, 0) frequency corner, which is where rfft2 places DC
    r = np.sqrt(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2)
    r_int = r.astype(np.int64)

    max_r = r_int.max()
    profile = np.zeros(max_r + 1, dtype=np.float64)
    counts = np.zeros(max_r + 1, dtype=np.float64)

    flat_r = r_int.ravel()
    flat_psd = psd_2d.ravel()

    np.add.at(profile, flat_r, flat_psd)
    np.add.at(counts, flat_r, 1.0)

    counts[counts == 0] = 1.0

    return profile / counts


@requires(spatial=True)
def psd_radial_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the L1 error between the radially-averaged power spectral
        density (PSD) of ground truth and prediction, averaged over batch,
        channel and time. More robust than the raw 2D `spectral_error` since
        it summarizes the spectrum by frequency magnitude rather than by
        direction, making it less sensitive to orientation artifacts.

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
            Mean L1 distance between radially-averaged power spectra. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import psd_radial_error

        gt   = np.random.rand(8, 64, 64, 1)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = psd_radial_error(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)
    y_true_c, y_pred_c = apply_nan_mask(y_true_c, y_pred_c, ignore_nan=ignore_nan)

    B, H, W, C, T = y_true_c.shape
    logger.debug("psd_radial_error: %d slices", B*C*T)
    errors = []
    for b in _progress(range(B), "psd_radial_error [batch]", B):
        for c in range(C):
            for t in range(T):
                true2d = y_true_c[b, :, :, c, t].astype(np.float64)
                pred2d = y_pred_c[b, :, :, c, t].astype(np.float64)

                true_psd = np.abs(np.fft.rfft2(true2d)) / (H * W)
                pred_psd = np.abs(np.fft.rfft2(pred2d)) / (H * W)

                true_profile = _radial_average(true_psd)
                pred_profile = _radial_average(pred_psd)

                n = min(len(true_profile), len(pred_profile))
                errors.append(np.mean(np.abs(true_profile[:n] - pred_profile[:n])))

    return float(np.mean(errors))


def spectral_angle_mapper(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Spectral Angle Mapper (SAM): the mean angle, in degrees,
        between the multi-band spectral signature vectors of ground truth
        and prediction at each pixel. Standard metric in earth observation
        for comparing multi-spectral/hyperspectral pixel signatures
        independently of overall brightness.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes, with channel
            (C) as the spectral-band axis.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are replaced with the finite median
            before computing the angle (default: False).

        Returns:
        --------
        - value : float
            Mean spectral angle in degrees, in [0, 180]. Lower is better
            (0 means identical spectral signature direction).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import spectral_angle_mapper

        gt   = np.random.rand(32, 32, 4)  # 4 spectral bands
        pred = gt * 1.05
        score = spectral_angle_mapper(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)
    y_true_c, y_pred_c = apply_nan_mask(y_true_c, y_pred_c, ignore_nan=ignore_nan)

    a = y_true_c.astype(np.float64)
    b = y_pred_c.astype(np.float64)

    # channel axis is index 3 in canonical (B, H, W, C, T)
    dot = np.sum(a * b, axis=3)
    norm_a = np.sqrt(np.sum(a ** 2, axis=3))
    norm_b = np.sqrt(np.sum(b ** 2, axis=3))

    denom = np.clip(norm_a * norm_b, a_min=1e-12, a_max=None)
    cos_angle = np.clip(dot / denom, -1.0, 1.0)

    angles_deg = np.degrees(np.arccos(cos_angle))

    return float(np.mean(angles_deg))
