from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DEFAULT_THRESHOLD


def validate_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Validate classification metric inputs."""
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Empty inputs are not allowed.")

    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("Predicted probabilities must be in [0, 1].")


def proba_to_labels(y_proba: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Convert probabilities into binary predictions."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")

    return (y_proba >= threshold).astype(int)


def classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """
    Compute core classification metrics for binary fraud detection.

    Returns:
        Dictionary with PR-AUC, ROC-AUC, F1, Precision, Recall, Specificity, Accuracy.
    """
    validate_inputs(y_true, y_proba)
    y_pred = proba_to_labels(y_proba, threshold=threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def confusion_counts(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, int]:
    """
    Return confusion matrix counts for a given threshold.
    """
    validate_inputs(y_true, y_proba)
    y_pred = proba_to_labels(y_proba, threshold=threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Full evaluation wrapper for a binary classifier.
    """
    metrics = classification_metrics(y_true, y_proba, threshold=threshold)
    counts = confusion_counts(y_true, y_proba, threshold=threshold)

    return {
        **metrics,
        **counts,
        "threshold": float(threshold),
    }
