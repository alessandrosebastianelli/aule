"""
    Ensemble validation metrics.

    These metrics expect an explicit ensemble dimension. Since the 4 standard
    aule shapes do not reserve an axis for ensemble members, all functions in
    this module take the ensemble as a separate argument: an array of shape
    (n_members, *single_member_shape), where single_member_shape is any of
    the 4 standard shapes (a/b/c/d) and ground truth has no ensemble axis.
"""

from typing import Optional
import numpy as np

from .._shapes import to_canonical, apply_nan_mask, finite_mask

__all__ = ["ensemble_spread", "crps", "rank_histogram"]


def ensemble_spread(
    y_ensemble: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the mean ensemble spread (standard deviation across members),
        averaged over all spatial/temporal positions. A measure of forecast
        uncertainty, independent of the ground truth.

        Parameters:
        -----------
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape), where
            single_member_shape is one of the 4 standard aule shapes.
        - data_format : str
            "bhwc" or "hwct", required only when each member is 4D.
        - ignore_nan : bool
            If True, non-finite values are excluded from the spread
            computation (default: False).

        Returns:
        --------
        - value : float
            Mean ensemble standard deviation. Larger values indicate
            higher forecast uncertainty.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import ensemble_spread

        ensemble = np.random.rand(10, 64, 64, 1)  # 10 members, (H, W, C)
        spread = ensemble_spread(ensemble)
        ```
    '''

    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)  # (M, B, H, W, C, T)

    if ignore_nan:
        return float(np.nanstd(np.where(np.isfinite(stacked), stacked, np.nan), axis=0).mean())

    return float(np.std(stacked, axis=0).mean())


def crps(
    y_true: np.ndarray,
    y_ensemble: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> float:
    '''
        Computes the Continuous Ranked Probability Score (CRPS) using the
        empirical ensemble distribution, via the standard pairwise-distance
        decomposition:

            CRPS = E|X - y| - 0.5 * E|X - X'|

        where X, X' are independent draws from the ensemble and y is the
        ground truth observation.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes (no ensemble axis).
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, non-finite values in y_true are replaced with the
            finite median before scoring (default: False).

        Returns:
        --------
        - value : float
            Mean CRPS over all positions. Lower is better (0 = perfect).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import crps

        gt       = np.random.rand(64, 64, 1)
        ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 64, 64, 1))
        score = crps(gt, ensemble)
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)  # (M, B, H, W, C, T)

    if ignore_nan:
        fill = float(np.nanmedian(np.where(np.isfinite(y_true_c), y_true_c, np.nan)))
        y_true_c = np.where(np.isfinite(y_true_c), y_true_c, fill)

    M = stacked.shape[0]
    y_true_b = y_true_c.astype(np.float64)[np.newaxis, ...]  # (1, B, H, W, C, T)

    # Term 1: E|X - y|
    term1 = np.mean(np.abs(stacked - y_true_b), axis=0)

    # Term 2: 0.5 * E|X - X'| over all member pairs (including i == j, standard convention)
    diffs = np.abs(stacked[:, np.newaxis, ...] - stacked[np.newaxis, :, ...])  # (M, M, ...)
    term2 = 0.5 * np.mean(diffs, axis=(0, 1))

    crps_field = term1 - term2

    return float(np.mean(crps_field))


def rank_histogram(
    y_true: np.ndarray,
    y_ensemble: np.ndarray,
    data_format: Optional[str] = None,
    ignore_nan: bool = False,
) -> np.ndarray:
    '''
        Computes the rank histogram (Talagrand diagram) counts: for each
        observation, the rank of the ground truth value relative to the
        sorted ensemble members. A flat histogram indicates a
        well-calibrated ensemble; U-shaped indicates under-dispersion;
        dome-shaped indicates over-dispersion.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes (no ensemble axis).
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.
        - ignore_nan : bool
            If True, positions with non-finite ground truth are excluded
            from the histogram (default: False).

        Returns:
        --------
        - counts : np.ndarray
            1D array of length (n_members + 1) with the count of observations
            falling into each rank bin.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import rank_histogram

        gt       = np.random.rand(64, 64, 1)
        ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 64, 64, 1))
        counts = rank_histogram(gt, ensemble)  # length 11
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)  # (M, B, H, W, C, T)
    M = stacked.shape[0]

    y_true_flat = y_true_c.astype(np.float64).ravel()
    members_flat = stacked.reshape(M, -1)  # (M, N)

    if ignore_nan:
        valid = np.isfinite(y_true_flat) & np.all(np.isfinite(members_flat), axis=0)
        y_true_flat = y_true_flat[valid]
        members_flat = members_flat[:, valid]

    # rank = number of ensemble members with value < observation
    ranks = np.sum(members_flat < y_true_flat[np.newaxis, :], axis=0)

    counts = np.bincount(ranks, minlength=M + 1)[: M + 1]

    return counts
