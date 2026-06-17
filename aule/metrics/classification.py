"""
    Classification and segmentation metrics for categorical/binary outputs.

    These metrics are designed to be as generic as possible: they accept
    either binary masks (values in {0, 1}) or multi-class integer label
    maps, controlled via the `average` parameter. Inputs follow the same 4
    standard aule shapes, with the channel axis (C) typically of size 1 for
    label maps.
"""

from typing import Optional
import numpy as np

from .._shapes import match_shapes

__all__ = ["iou", "dice", "precision_recall_f1", "confusion_matrix_metrics", "cohen_kappa"]


def _unique_classes(y_true: np.ndarray, y_pred: np.ndarray, num_classes: Optional[int]) -> np.ndarray:
    '''
        Determines the set of class labels to evaluate.

        Parameters:
        -----------
        - y_true, y_pred : np.ndarray
            Integer label arrays.
        - num_classes : int
            If given, classes are assumed to be 0..num_classes-1. Otherwise
            inferred as the sorted union of unique values in both arrays.

        Returns:
        --------
        - classes : np.ndarray
            1D array of class labels to evaluate.
    '''

    if num_classes is not None:
        return np.arange(num_classes)
    return np.unique(np.concatenate([y_true.ravel(), y_pred.ravel()]))


def iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    average: str = "binary",
    data_format: Optional[str] = None,
) -> float:
    '''
        Computes the Intersection over Union (Jaccard index) between ground
        truth and predicted masks/label maps. Works for binary masks
        (values in {0, 1}, `average="binary"`) or multi-class integer label
        maps (`average="macro"` or `"micro"`).

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth mask or label map, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Predicted mask or label map, same shape as y_true.
        - num_classes : int
            Number of classes for multi-class averaging. If None, inferred
            from the unique values present in y_true/y_pred.
        - average : str
            "binary" treats inputs as a single foreground class (value 1, or
            any nonzero value); "macro" computes per-class IoU and averages
            unweighted; "micro" pools all classes' intersection/union before
            dividing (default: "binary").
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.

        Returns:
        --------
        - value : float
            IoU score in [0, 1]. Higher is better. Returns 1.0 when both
            masks are empty (no foreground in either).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import iou

        gt   = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        pred = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        score = iou(gt, pred)

        # multi-class
        gt_labels   = np.random.randint(0, 4, (64, 64, 1))
        pred_labels = np.random.randint(0, 4, (64, 64, 1))
        score_macro = iou(gt_labels, pred_labels, num_classes=4, average="macro")
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    if average == "binary":
        a = (y_true_c != 0)
        b = (y_pred_c != 0)
        intersection = np.sum(a & b)
        union = np.sum(a | b)
        if union == 0:
            return 1.0
        return float(intersection / union)

    classes = _unique_classes(y_true_c, y_pred_c, num_classes)

    if average == "micro":
        total_intersection, total_union = 0.0, 0.0
        for cls in classes:
            a = (y_true_c == cls)
            b = (y_pred_c == cls)
            total_intersection += np.sum(a & b)
            total_union += np.sum(a | b)
        if total_union == 0:
            return 1.0
        return float(total_intersection / total_union)

    if average == "macro":
        scores = []
        for cls in classes:
            a = (y_true_c == cls)
            b = (y_pred_c == cls)
            union = np.sum(a | b)
            scores.append(1.0 if union == 0 else float(np.sum(a & b) / union))
        return float(np.mean(scores))

    raise ValueError("average must be 'binary', 'macro', or 'micro'")


def dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    average: str = "binary",
    data_format: Optional[str] = None,
) -> float:
    '''
        Computes the Dice coefficient (F1 score over set overlap) between
        ground truth and predicted masks/label maps. Works for binary masks
        or multi-class integer label maps, mirroring `iou`'s `average` options.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth mask or label map, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Predicted mask or label map, same shape as y_true.
        - num_classes : int
            Number of classes for multi-class averaging. If None, inferred
            from the unique values present in y_true/y_pred.
        - average : str
            "binary", "macro", or "micro" (see `iou`) (default: "binary").
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.

        Returns:
        --------
        - value : float
            Dice score in [0, 1]. Higher is better. Returns 1.0 when both
            masks are empty (no foreground in either).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import dice

        gt   = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        pred = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        score = dice(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    def _dice_for_class(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.sum(a) + np.sum(b)
        if denom == 0:
            return 1.0
        return float(2.0 * np.sum(a & b) / denom)

    if average == "binary":
        a = (y_true_c != 0)
        b = (y_pred_c != 0)
        return _dice_for_class(a, b)

    classes = _unique_classes(y_true_c, y_pred_c, num_classes)

    if average == "micro":
        total_inter, total_denom = 0.0, 0.0
        for cls in classes:
            a = (y_true_c == cls)
            b = (y_pred_c == cls)
            total_inter += np.sum(a & b)
            total_denom += np.sum(a) + np.sum(b)
        if total_denom == 0:
            return 1.0
        return float(2.0 * total_inter / total_denom)

    if average == "macro":
        scores = [_dice_for_class(y_true_c == cls, y_pred_c == cls) for cls in classes]
        return float(np.mean(scores))

    raise ValueError("average must be 'binary', 'macro', or 'micro'")


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    average: str = "binary",
    data_format: Optional[str] = None,
) -> dict:
    '''
        Computes precision, recall and F1 score between ground truth and
        predicted masks/label maps. Works for binary masks or multi-class
        integer label maps.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth mask or label map, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Predicted mask or label map, same shape as y_true.
        - num_classes : int
            Number of classes for multi-class averaging. If None, inferred
            from the unique values present in y_true/y_pred.
        - average : str
            "binary" treats nonzero as the positive class; "macro" averages
            per-class scores unweighted; "micro" pools true/false positives
            and negatives across classes before computing the scores
            (default: "binary").
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.

        Returns:
        --------
        - scores : dict
            Dictionary with keys "precision", "recall", "f1", each in [0, 1].

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import precision_recall_f1

        gt   = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        pred = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        scores = precision_recall_f1(gt, pred)
        print(scores["precision"], scores["recall"], scores["f1"])
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    def _prf(tp: float, fp: float, fn: float) -> dict:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    if average == "binary":
        a = (y_true_c != 0)
        b = (y_pred_c != 0)
        tp = float(np.sum(a & b))
        fp = float(np.sum(~a & b))
        fn = float(np.sum(a & ~b))
        return _prf(tp, fp, fn)

    classes = _unique_classes(y_true_c, y_pred_c, num_classes)

    if average == "micro":
        tp_total, fp_total, fn_total = 0.0, 0.0, 0.0
        for cls in classes:
            a = (y_true_c == cls)
            b = (y_pred_c == cls)
            tp_total += np.sum(a & b)
            fp_total += np.sum(~a & b)
            fn_total += np.sum(a & ~b)
        return _prf(tp_total, fp_total, fn_total)

    if average == "macro":
        results = []
        for cls in classes:
            a = (y_true_c == cls)
            b = (y_pred_c == cls)
            tp = float(np.sum(a & b))
            fp = float(np.sum(~a & b))
            fn = float(np.sum(a & ~b))
            results.append(_prf(tp, fp, fn))
        return {
            "precision": float(np.mean([r["precision"] for r in results])),
            "recall": float(np.mean([r["recall"] for r in results])),
            "f1": float(np.mean([r["f1"] for r in results])),
        }

    raise ValueError("average must be 'binary', 'macro', or 'micro'")


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    data_format: Optional[str] = None,
) -> dict:
    '''
        Computes standard binary confusion-matrix-derived metrics (accuracy,
        specificity, balanced accuracy) for a binary mask comparison.
        For multi-class evaluation, use `precision_recall_f1` with
        `average="macro"` or `"micro"` instead.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth binary mask, any of the 4 supported shapes
            (any nonzero value is treated as positive).
        - y_pred : np.ndarray
            Predicted binary mask, same shape as y_true.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.

        Returns:
        --------
        - metrics : dict
            Dictionary with keys "accuracy", "specificity", "balanced_accuracy",
            "tp", "fp", "tn", "fn" (the latter four as raw counts).

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import confusion_matrix_metrics

        gt   = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        pred = (np.random.rand(64, 64, 1) > 0.5).astype(int)
        metrics = confusion_matrix_metrics(gt, pred)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    a = (y_true_c != 0)
    b = (y_pred_c != 0)

    tp = float(np.sum(a & b))
    fp = float(np.sum(~a & b))
    tn = float(np.sum(~a & ~b))
    fn = float(np.sum(a & ~b))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def cohen_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    data_format: Optional[str] = None,
) -> float:
    '''
        Computes Cohen's Kappa coefficient, a chance-corrected agreement
        score between ground truth and predicted label maps. Works for
        binary or multi-class integer labels.

        Parameters:
        -----------
        - y_true : np.ndarray
            Ground truth label map, any of the 4 supported shapes.
        - y_pred : np.ndarray
            Predicted label map, same shape as y_true.
        - num_classes : int
            Number of classes. If None, inferred from the unique values
            present in y_true/y_pred.
        - data_format : str
            "bhwc" or "hwct", required only when arrays are 4D.

        Returns:
        --------
        - value : float
            Cohen's Kappa, typically in [-1, 1]. 1.0 is perfect agreement,
            0.0 is agreement no better than chance.

        Usage:
        ------

        ```python
        import numpy as np
        from aule.metrics import cohen_kappa

        gt   = np.random.randint(0, 4, (64, 64, 1))
        pred = np.random.randint(0, 4, (64, 64, 1))
        score = cohen_kappa(gt, pred, num_classes=4)
        ```
    '''

    y_true_c, y_pred_c = match_shapes(y_true, y_pred, data_format=data_format)

    classes = _unique_classes(y_true_c, y_pred_c, num_classes)
    n_classes = len(classes)

    a_flat = y_true_c.ravel()
    b_flat = y_pred_c.ravel()
    n = a_flat.size

    # map class labels to dense indices 0..n_classes-1 via searchsorted
    sorted_classes = np.sort(classes)
    true_idx = np.searchsorted(sorted_classes, a_flat)
    pred_idx = np.searchsorted(sorted_classes, b_flat)

    valid = (true_idx < n_classes) & (pred_idx < n_classes) & \
            (sorted_classes[np.clip(true_idx, 0, n_classes - 1)] == a_flat) & \
            (sorted_classes[np.clip(pred_idx, 0, n_classes - 1)] == b_flat)

    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(confusion, (true_idx[valid], pred_idx[valid]), 1.0)

    observed_agreement = np.trace(confusion) / n

    row_marginals = confusion.sum(axis=1) / n
    col_marginals = confusion.sum(axis=0) / n
    expected_agreement = np.sum(row_marginals * col_marginals)

    if expected_agreement == 1.0:
        return 1.0

    return float((observed_agreement - expected_agreement) / (1.0 - expected_agreement))
