import sys
sys.path += ['.']

import numpy as np
import pytest

from aule.metrics import fractions_skill_score, energy_score


def test_fss_perfect_for_identical():
    x = np.random.exponential(1.0, (32, 32, 1))
    assert fractions_skill_score(x, x, threshold=1.0, window=9) == pytest.approx(1.0, abs=1e-10)


def test_fss_bounded():
    gt = np.random.exponential(1.0, (32, 32, 1))
    pred = gt + np.random.normal(0, 0.5, gt.shape)
    score = fractions_skill_score(gt, pred, threshold=1.0, window=9)
    assert 0.0 <= score <= 1.0


def test_fss_no_events_returns_one():
    gt = np.zeros((16, 16, 1))
    pred = np.zeros((16, 16, 1))
    assert fractions_skill_score(gt, pred, threshold=5.0, window=5) == pytest.approx(1.0, abs=1e-10)


def test_fss_tolerates_small_displacement_better_than_no_window():
    rng = np.random.default_rng(0)
    gt = np.zeros((32, 32, 1))
    gt[10:15, 10:15] = 2.0  # a small "event" blob

    pred_shifted = np.zeros((32, 32, 1))
    pred_shifted[11:16, 11:16] = 2.0  # same blob, shifted by 1 pixel

    score_small_window = fractions_skill_score(gt, pred_shifted, threshold=1.0, window=3)
    score_large_window = fractions_skill_score(gt, pred_shifted, threshold=1.0, window=15)

    # a larger neighborhood window should be at least as forgiving of the displacement
    assert score_large_window >= score_small_window - 1e-6


def test_energy_score_zero_for_perfect_ensemble():
    gt = np.random.rand(16, 16, 1)
    ensemble = np.stack([gt] * 10, axis=0)
    assert energy_score(gt, ensemble) == pytest.approx(0.0, abs=1e-6)


def test_energy_score_nonnegative():
    gt = np.random.rand(16, 16, 1)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    assert energy_score(gt, ensemble) >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
