from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


EPS = 1e-15


def validate_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Validate probabilistic scoring inputs."""
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Empty inputs are not allowed.")

    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("Predicted probabilities must be in [0, 1].")


def clip_probabilities(y_proba: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip probabilities to avoid log(0)."""
    return np.clip(y_proba, eps, 1.0 - eps)


def negative_log_likelihood(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute binary negative log-likelihood (log loss).

    Lower is better.
    """
    validate_inputs(y_true, y_proba)
    y_proba = clip_probabilities(y_proba)
    return float(log_loss(y_true, y_proba))


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Brier score for binary classification.

    Lower is better.
    """
    validate_inputs(y_true, y_proba)
    return float(brier_score_loss(y_true, y_proba))


def log_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Alias for average log score under the Bernoulli model.

    Here returned as negative log-likelihood for consistency in minimization.
    Lower is better.
    """
    return negative_log_likelihood(y_true, y_proba)


def probabilistic_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """
    Compute core proper scoring rules for binary classification.

    Returns:
        Dictionary with NLL and Brier score.
    """
    return {
        "nll": negative_log_likelihood(y_true, y_proba),
        "brier_score": brier_score(y_true, y_proba),
        "log_score": log_score(y_true, y_proba),
    }