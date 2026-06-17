import sys
sys.path += ['.']

import numpy as np
import pytest

from aule.metrics import picp, pit_histogram


def test_picp_one_for_wide_ensemble():
    gt = np.random.rand(16, 16, 1)
    # extremely wide ensemble should cover virtually all observations
    ensemble = gt[np.newaxis] + np.random.normal(0, 5.0, (20, 16, 16, 1))
    coverage = picp(gt, ensemble, confidence=0.9)
    assert coverage > 0.85


def test_picp_bounded():
    gt = np.random.rand(16, 16, 1)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    coverage = picp(gt, ensemble, confidence=0.9)
    assert 0.0 <= coverage <= 1.0


def test_picp_zero_when_ensemble_far_from_truth():
    gt = np.zeros((16, 16, 1))
    ensemble = np.full((10, 16, 16, 1), 100.0)
    coverage = picp(gt, ensemble, confidence=0.9)
    assert coverage == pytest.approx(0.0, abs=1e-10)


def test_pit_histogram_length():
    gt = np.random.rand(16, 16, 1)
    ensemble = gt[np.newaxis] + np.random.normal(0, 0.1, (10, 16, 16, 1))
    counts = pit_histogram(gt, ensemble, n_bins=10)
    assert len(counts) == 10
    assert counts.sum() == gt.size


def test_pit_histogram_roughly_uniform_for_well_calibrated_ensemble():
    rng = np.random.default_rng(1)
    # ensemble members drawn from the same generative process as the truth
    n_members = 50
    truth_and_members = rng.normal(0, 1, (n_members + 1, 64, 64, 1))
    gt = truth_and_members[0]
    ensemble = truth_and_members[1:]
    counts = pit_histogram(gt, ensemble, n_bins=10)
    expected = counts.sum() / 10
    # bins shouldn't be wildly skewed for a well-calibrated case
    assert np.max(counts) < expected * 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
