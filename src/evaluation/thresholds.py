import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def find_best_f1_threshold(y_true, y_proba):
    """
    Find threshold that maximizes F1 score.

    Uses thresholds derived from precision-recall curve
    instead of uniform grid search.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Edge case: sklearn returns thresholds of size len-1
    thresholds = np.append(thresholds, 1.0)

    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return float(best_thr), float(best_f1)


def find_threshold_for_target_recall(y_true, y_proba, target_recall=0.9):
    """
    Find the highest threshold that achieves at least target recall.

    We iterate from high threshold → low threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Align shapes
    thresholds = np.append(thresholds, 1.0)

    # Reverse (high threshold → low threshold)
    for r, t in zip(recall[::-1], thresholds[::-1]):
        if r >= target_recall:
            return float(t)

    # Fallback
    return 0.5


def find_threshold_for_target_precision(y_true, y_proba, target_precision=0.9):
    """
    Find the lowest threshold that achieves at least target precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    thresholds = np.append(thresholds, 1.0)

    for p, t in zip(precision, thresholds):
        if p >= target_precision:
            return float(t)

    return 0.5


def compute_threshold_metrics(y_true, y_proba, thresholds=None):
    """
    Evaluate multiple thresholds at once.

    Useful for analysis / plotting / decision design.
    """
    if thresholds is None:
        # Use unique probability values (best practice)
        thresholds = np.unique(y_proba)

    results = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return results