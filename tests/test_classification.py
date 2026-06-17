import sys
sys.path += ['.']

import numpy as np
import pytest

from aule.metrics import iou, dice, precision_recall_f1, confusion_matrix_metrics, cohen_kappa


def test_iou_binary_perfect_for_identical():
    x = (np.random.rand(32, 32, 1) > 0.5).astype(int)
    assert iou(x, x) == pytest.approx(1.0, abs=1e-10)


def test_iou_binary_zero_for_disjoint_masks():
    a = np.zeros((4, 4, 1), dtype=int)
    a[:2, :2] = 1
    b = np.zeros((4, 4, 1), dtype=int)
    b[2:, 2:] = 1
    assert iou(a, b) == pytest.approx(0.0, abs=1e-10)


def test_iou_empty_masks_returns_one():
    a = np.zeros((8, 8, 1), dtype=int)
    b = np.zeros((8, 8, 1), dtype=int)
    assert iou(a, b) == pytest.approx(1.0, abs=1e-10)


def test_iou_macro_perfect_for_identical_multiclass():
    x = np.random.randint(0, 4, (32, 32, 1))
    assert iou(x, x, num_classes=4, average="macro") == pytest.approx(1.0, abs=1e-10)


def test_iou_micro_perfect_for_identical_multiclass():
    x = np.random.randint(0, 4, (32, 32, 1))
    assert iou(x, x, num_classes=4, average="micro") == pytest.approx(1.0, abs=1e-10)


def test_dice_binary_perfect_for_identical():
    x = (np.random.rand(32, 32, 1) > 0.5).astype(int)
    assert dice(x, x) == pytest.approx(1.0, abs=1e-10)


def test_dice_at_least_iou():
    a = (np.random.rand(32, 32, 1) > 0.5).astype(int)
    b = (np.random.rand(32, 32, 1) > 0.5).astype(int)
    assert dice(a, b) >= iou(a, b) - 1e-10


def test_precision_recall_f1_perfect_for_identical():
    x = (np.random.rand(32, 32, 1) > 0.5).astype(int)
    result = precision_recall_f1(x, x)
    assert result["precision"] == pytest.approx(1.0, abs=1e-10)
    assert result["recall"] == pytest.approx(1.0, abs=1e-10)
    assert result["f1"] == pytest.approx(1.0, abs=1e-10)


def test_precision_recall_f1_macro_multiclass():
    x = np.random.randint(0, 3, (16, 16, 1))
    result = precision_recall_f1(x, x, num_classes=3, average="macro")
    assert result["f1"] == pytest.approx(1.0, abs=1e-10)


def test_confusion_matrix_metrics_perfect_for_identical():
    x = (np.random.rand(32, 32, 1) > 0.5).astype(int)
    result = confusion_matrix_metrics(x, x)
    assert result["accuracy"] == pytest.approx(1.0, abs=1e-10)
    assert result["balanced_accuracy"] == pytest.approx(1.0, abs=1e-10)
    assert result["fp"] == 0.0
    assert result["fn"] == 0.0


def test_confusion_matrix_metrics_counts_sum_to_total():
    a = (np.random.rand(16, 16, 1) > 0.5).astype(int)
    b = (np.random.rand(16, 16, 1) > 0.5).astype(int)
    result = confusion_matrix_metrics(a, b)
    assert result["tp"] + result["fp"] + result["tn"] + result["fn"] == a.size


def test_cohen_kappa_perfect_for_identical():
    x = np.random.randint(0, 4, (32, 32, 1))
    assert cohen_kappa(x, x, num_classes=4) == pytest.approx(1.0, abs=1e-10)


def test_cohen_kappa_near_zero_for_random_labels():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, (200, 200, 1))
    b = rng.integers(0, 4, (200, 200, 1))
    score = cohen_kappa(a, b, num_classes=4)
    assert abs(score) < 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
