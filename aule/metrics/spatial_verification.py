"""
    Spatial and multivariate verification metrics that go beyond simple
    pixel-wise comparison: neighborhood-based skill scores that tolerate
    small spatial displacement, and multivariate generalizations of
    ensemble scoring rules.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes, to_canonical
from .._guards import requires

__all__ = ["fractions_skill_score", "energy_score"]


def _box_fraction_field(binary_field: np.ndarray, window: int) -> np.ndarray:
    '''
        Computes, for each pixel, the fraction of "on" pixels within a
        square neighborhood of side `window`, using a cumulative-sum based
        box filter (no SciPy dependency needed).

        Parameters:
        -----------
        - binary_field : np.ndarray
            2D boolean/float array (H, W) with values in {0, 1}.
        - window : int
            Side length of the square neighborhood (must be odd; if even,
            it is incremented by 1).

        Returns:
        --------
        - fractions : np.ndarray
            2D array (H, W), the local neighborhood fraction at each pixel.
    '''

    if window % 2 == 0:
        window += 1
    pad = window // 2

    padded = np.pad(binary_field.astype(np.float64), pad, mode="constant", constant_values=0.0)

    cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumsum = np.pad(cumsum, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

    H, W = binary_field.shape
    box_sum = (
        cumsum[window:window + H, window:window + W]
        - cumsum[0:H, window:window + W]
        - cumsum[window:window + H, 0:W]
        + cumsum[0:H, 0:W]
    )

    return box_sum / float(window * window)


@requires(spatial=True)
def fractions_skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    window: int = 9,
    batch_index: int = 0,
    channel_index: int = 0,
    time_index: int = 0,
    data_format: Optional[str] = None,
) -> float:
    '''
        Computes the Fractions Skill Score (FSS) at a given event threshold
        and neighborhood window size, for a single (batch, channel, time)
        slice. FSS compares the local neighborhood fraction of "event"
        pixels (value > threshold) between ground truth and prediction,
        making it tolerant of small spatial displacement errors that would
        otherwise heavily penalize a pixel-wise score like IoU. Standard
        verification metric for precipitation nowcasting and other
        spatially-displaced forecast fields.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Prediction array, same shape as y_true.
        - threshold : float
            Threshold defining the binary event (value > threshold).
        - window : int
            Side length (in pixels) of the square neighborhood used to
            compute local event fractions (default: 9). Larger windows
            tolerate larger displacement errors.
        - batch_index, channel_index, time_index : int
            Which slice to evaluate, when the corresponding axis has size > 1.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.

        Returns:
        --------
        - value : float
            FSS in [0, 1]. 1.0 is a perfect forecast; values below ~0.5 +
            (event frequency / 2) are generally considered "no skill"
            relative to a uniform-probability forecast.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import fractions_skill_score

        gt   = np.random.exponential(1.0, (64, 64, 1))
        pred = gt + np.random.normal(0, 0.3, gt.shape)
        score = fractions_skill_score(gt, pred, threshold=1.0, window=9)
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    y_pred_c = to_canonical(y_pred, data_format=data_format)

    true_field = y_true_c[batch_index, :, :, channel_index, time_index].astype(np.float64)
    pred_field = y_pred_c[batch_index, :, :, channel_index, time_index].astype(np.float64)

    true_binary = (true_field > threshold).astype(np.float64)
    pred_binary = (pred_field > threshold).astype(np.float64)

    true_fractions = _box_fraction_field(true_binary, window)
    pred_fractions = _box_fraction_field(pred_binary, window)

    mse_fractions = float(np.mean((true_fractions - pred_fractions) ** 2))
    mse_reference = float(np.mean(true_fractions ** 2) + np.mean(pred_fractions ** 2))

    if mse_reference == 0:
        # no event pixels anywhere in either field within any neighborhood
        return 1.0

    return float(1.0 - mse_fractions / mse_reference)


def energy_score(
    y_true: np.ndarray,
    y_ensemble: np.ndarray,
    data_format: Optional[str] = None,
) -> float:
    '''
        Computes the Energy Score, the multivariate generalization of CRPS,
        treating each spatial/channel position at a fixed (batch, time)
        index as one component of a joint multivariate forecast vector.
        Unlike `aule.metrics.crps` (which scores each pixel independently
        and ignores cross-pixel/cross-channel dependence), the energy score
        rewards ensembles that correctly capture the joint structure
        (e.g. spatial coherence, inter-band correlation) of the forecast,
        not just marginal calibration at each point.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth array, any of the 4 supported shapes (no ensemble axis).
        - y_ensemble : np.ndarray
            Ensemble array of shape (n_members, *single_member_shape).
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.

        Returns:
        --------
        - value : float
            Mean energy score over batch and time. Lower is better (0 = perfect).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import energy_score

        gt       = np.random.rand(32, 32, 1)
        ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (20, 32, 32, 1))
        score = energy_score(gt, ensemble)
        ```
    '''

    y_true_c = to_canonical(y_true, data_format=data_format)
    canonical_members = [to_canonical(member, data_format=data_format) for member in y_ensemble]
    stacked = np.stack(canonical_members, axis=0).astype(np.float64)  # (M, B, H, W, C, T)

    M, B, H, W, C, T = stacked.shape

    # flatten spatial/channel dims into a single "vector" axis per (batch, time) sample
    members_flat = stacked.transpose(1, 5, 0, 2, 3, 4).reshape(B, T, M, H * W * C)
    truth_flat = y_true_c.astype(np.float64).transpose(0, 4, 1, 2, 3).reshape(B, T, H * W * C)

    scores = np.zeros((B, T), dtype=np.float64)
    for b in range(B):
        for t in range(T):
            members = members_flat[b, t]  # (M, D)
            truth_vec = truth_flat[b, t]   # (D,)

            term1 = np.mean(np.linalg.norm(members - truth_vec[np.newaxis, :], axis=1))

            diffs = members[:, np.newaxis, :] - members[np.newaxis, :, :]  # (M, M, D)
            term2 = 0.5 * np.mean(np.linalg.norm(diffs, axis=2))

            scores[b, t] = term1 - term2

    return float(np.mean(scores))
