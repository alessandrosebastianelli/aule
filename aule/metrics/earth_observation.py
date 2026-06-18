"""
    Earth-observation specific metrics.

    These cover two common EO validation needs: comparing normalized
    difference indices (e.g. NDVI, NDWI, NDSI - any index of the form
    (band_a - band_b) / (band_a + band_b)) computed from predicted vs
    ground truth bands, and quantifying multi-temporal change detection
    error when a time axis is present.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes, to_canonical
from .._guards import requires

__all__ = ["normalized_difference_index", "index_error", "change_detection_error"]


def normalized_difference_index(
    band_a: np.ndarray,
    band_b: np.ndarray,
    data_format: Optional[str] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    '''
        Computes a generic normalized difference index: (band_a - band_b) / (band_a + band_b).
        This covers NDVI (NIR, Red), NDWI (Green, NIR), NDSI (Green, SWIR) and
        similar two-band indices commonly used in earth observation.

        Parameters:
        -----------
        - band_a : np.ndarray
            First band array, any of the 4 supported shapes.
        - band_b : np.ndarray
            Second band array, same shape as band_a.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - eps : float
            Small constant added to the denominator to avoid division by
            zero (default: 1e-6).

        Returns:
        --------
        - index : np.ndarray
            Normalized difference index, same canonical shape as the inputs,
            with values in [-1, 1].

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import normalized_difference_index

        nir = np.random.rand(64, 64, 1) * 0.5 + 0.3
        red = np.random.rand(64, 64, 1) * 0.3 + 0.1
        ndvi = normalized_difference_index(nir, red)
        ```
    '''

    a_c, b_c = match_shapes(band_a, band_b, data_format=data_format)
    a_c, b_c = a_c.astype(np.float64), b_c.astype(np.float64)

    return (a_c - b_c) / (a_c + b_c + eps)


def index_error(
    true_band_a: np.ndarray,
    true_band_b: np.ndarray,
    pred_band_a: np.ndarray,
    pred_band_b: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
    eps: float = 1e-6,
) -> float:
    '''
        Computes the RMSE between a normalized difference index (e.g. NDVI)
        derived from ground truth bands and the same index derived from
        predicted bands. Useful when the downstream use case cares about
        the index itself rather than the raw reflectance bands.

        Parameters:
        -----------
        - true_band_a, true_band_b : np.ndarray
            Ground truth bands, any of the 4 supported shapes.
        - pred_band_a, pred_band_b : np.ndarray
            Predicted bands, same shape as the ground truth bands.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the RMSE (default: False).
        - eps : float
            Small constant added to the denominator to avoid division by zero.

        Returns:
        --------
        - value : float
            RMSE between the two derived indices. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import index_error

        true_nir, true_red = np.random.rand(64,64,1), np.random.rand(64,64,1)
        pred_nir, pred_red = true_nir + 0.02, true_red - 0.01
        score = index_error(true_nir, true_red, pred_nir, pred_red)
        ```
    '''

    true_index = normalized_difference_index(true_band_a, true_band_b, data_format=data_format, eps=eps)
    pred_index = normalized_difference_index(pred_band_a, pred_band_b, data_format=data_format, eps=eps)

    diff = true_index - pred_index
    if ignore_nan:
        mask = np.isfinite(diff)
        diff = diff[mask]

    return float(np.sqrt(np.mean(diff ** 2)))


@requires(temporal=True)
def change_detection_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the RMSE between the ground truth and predicted temporal
        differences (i.e. how well the model reproduces change between
        consecutive time steps, rather than absolute values). Requires a
        time axis with at least 2 steps.

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
            If True, non-finite values are excluded from the RMSE (default: False).

        Returns:
        --------
        - value : float
            RMSE between consecutive-step differences. Lower is better.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import change_detection_error

        gt   = np.random.rand(64, 64, 1, 12)
        pred = gt + np.random.normal(0, 0.05, gt.shape)
        score = change_detection_error(gt, pred, data_format="hwct")
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    if y_true_c.shape[-1] < 2:
        raise ValueError(
            "change_detection_error requires at least 2 time steps; "
            f"got T={y_true_c.shape[-1]}. Did you forget data_format='hwct'?"
        )

    true_diff = np.diff(y_true_c.astype(np.float64), axis=-1)
    pred_diff = np.diff(y_pred_c.astype(np.float64), axis=-1)

    error = true_diff - pred_diff

    if ignore_nan:
        mask = np.isfinite(error)
        error = error[mask]

    return float(np.sqrt(np.mean(error ** 2)))
