"""
    Uncertainty calibration metrics for probabilistic/ensemble forecasts.

    These metrics assess whether predicted uncertainty (e.g. an ensemble
    spread or prediction interval) is well calibrated, independent of the
    point-forecast accuracy already covered by `aule.metrics.ensemble`.
"""

from typing import Optional
import numpy as np

from .._shapes import to_canonical

__all__ = ["picp", "pit_histogram"]


def picp(
    y_true: np.ndarray,
    y_ensemble: np.ndarray,
    confidence: float = 0.9,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Prediction Interval Coverage Probability (PICP): the
        fraction of ground truth observations that fall within the
        empirical prediction interval derived from the ensemble, at a given
        confidence level. A well-calibrated ensemble has PICP close to the
        nominal `confidence` level.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes (no ensemble axis).
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape).
        - confidence : float
            Nominal confidence level of the interval, in (0, 1) (default: 0.9,
            i.e. a 90% interval using the 5th/95th percentiles).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite ground truth positions are excluded (default: False).

        Returns:
        --------
        - value : float
            Coverage probability in [0, 1]. Should be close to `confidence`
            for a well-calibrated ensemble; much higher means the interval
            is too wide (over-dispersive), much lower means too narrow
            (under-dispersive).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import picp

        gt       = np.random.rand(64, 64, 1)
        ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 64, 64, 1))
        coverage = picp(gt, ensemble, confidence=0.9)
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)  # (M, B, H, W, C, T)

    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(stacked, 100 * alpha, axis=0)
    upper = np.percentile(stacked, 100 * (1 - alpha), axis=0)

    observed = y_true_c.astype(np.float64)
    inside = (observed >= lower) & (observed <= upper)

    if ignore_nan:
        mask = np.isfinite(observed)
        inside = inside[mask]

    return float(np.mean(inside))


def pit_histogram(
    y_true: np.ndarray,
    y_ensemble: np.ndarray,
    n_bins: int = 10,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> np.ndarray:
    '''
        Computes the Probability Integral Transform (PIT) histogram: for
        each observation, the empirical CDF value of the ensemble at the
        observed value, binned into a histogram. A well-calibrated
        probabilistic forecast produces a flat (uniform) PIT histogram;
        U-shaped indicates under-dispersion, dome-shaped over-dispersion,
        and skew indicates systematic bias. Closely related to
        `aule.metrics.rank_histogram` but using a continuous CDF estimate
        rather than discrete ranks.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes (no ensemble axis).
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape).
        - n_bins : int
            Number of histogram bins over [0, 1] (default: 10).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite ground truth positions are excluded (default: False).

        Returns:
        --------
        - counts : np.ndarray
            1D array of length `n_bins` with the histogram counts.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import pit_histogram

        gt       = np.random.rand(64, 64, 1)
        ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 64, 64, 1))
        counts = pit_histogram(gt, ensemble, n_bins=10)
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)  # (M, B, H, W, C, T)
    M = stacked.shape[0]

    observed = y_true_c.astype(np.float64).ravel()
    members_flat = stacked.reshape(M, -1)

    if ignore_nan:
        valid = np.isfinite(observed)
        observed = observed[valid]
        members_flat = members_flat[:, valid]

    # empirical CDF estimate with small random jitter within rank ties,
    # following the standard PIT randomization for discrete ensembles
    rank_below = np.sum(members_flat < observed[np.newaxis, :], axis=0)
    rank_equal = np.sum(members_flat == observed[np.newaxis, :], axis=0)

    rng = np.random.default_rng(42)
    pit_values = (rank_below + rng.uniform(0.0, 1.0, size=observed.shape) * (rank_equal + 1)) / (M + 1)
    pit_values = np.clip(pit_values, 0.0, 1.0)

    counts, _ = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))

    return counts
