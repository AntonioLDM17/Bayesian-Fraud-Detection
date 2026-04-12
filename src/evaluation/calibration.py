from __future__ import annotations

from typing import Any

import numpy as np


def validate_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Validate calibration inputs."""
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Empty inputs are not allowed.")

    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("Predicted probabilities must be in [0, 1].")


def calibration_bins(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Compute calibration statistics by probability bins.

    Returns:
        Dictionary containing:
        - bin_edges
        - bin_centers
        - bin_counts
        - mean_confidence
        - empirical_accuracy
    """
    validate_inputs(y_true, y_proba)

    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bin_edges[1:-1], right=True)

    bin_counts = np.zeros(n_bins, dtype=int)
    mean_confidence = np.zeros(n_bins, dtype=float)
    empirical_accuracy = np.zeros(n_bins, dtype=float)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(np.sum(mask))
        bin_counts[bin_idx] = count

        if count > 0:
            mean_confidence[bin_idx] = float(np.mean(y_proba[mask]))
            empirical_accuracy[bin_idx] = float(np.mean(y_true[mask]))
        else:
            mean_confidence[bin_idx] = 0.0
            empirical_accuracy[bin_idx] = 0.0

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_counts": bin_counts,
        "mean_confidence": mean_confidence,
        "empirical_accuracy": empirical_accuracy,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum over bins of:
        (bin_count / total_count) * |accuracy_bin - confidence_bin|
    """
    stats = calibration_bins(y_true, y_proba, n_bins=n_bins)

    total_count = np.sum(stats["bin_counts"])
    if total_count == 0:
        return 0.0

    ece = 0.0
    for count, acc, conf in zip(
        stats["bin_counts"],
        stats["empirical_accuracy"],
        stats["mean_confidence"],
    ):
        if count > 0:
            ece += (count / total_count) * abs(acc - conf)

    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE = max over bins of:
        |accuracy_bin - confidence_bin|
    """
    stats = calibration_bins(y_true, y_proba, n_bins=n_bins)

    errors = []
    for count, acc, conf in zip(
        stats["bin_counts"],
        stats["empirical_accuracy"],
        stats["mean_confidence"],
    ):
        if count > 0:
            errors.append(abs(acc - conf))

    return float(max(errors)) if errors else 0.0


def calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute core calibration metrics.
    """
    return {
        "ece": expected_calibration_error(y_true, y_proba, n_bins=n_bins),
        "mce": maximum_calibration_error(y_true, y_proba, n_bins=n_bins),
    }


def calibration_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """
    Return calibration statistics as a list of dictionaries,
    useful for saving to JSON or converting to a DataFrame.
    """
    stats = calibration_bins(y_true, y_proba, n_bins=n_bins)

    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        rows.append(
            {
                "bin_index": i,
                "bin_left": float(stats["bin_edges"][i]),
                "bin_right": float(stats["bin_edges"][i + 1]),
                "bin_center": float(stats["bin_centers"][i]),
                "count": int(stats["bin_counts"][i]),
                "mean_confidence": float(stats["mean_confidence"][i]),
                "empirical_accuracy": float(stats["empirical_accuracy"][i]),
                "gap": float(
                    abs(
                        stats["empirical_accuracy"][i]
                        - stats["mean_confidence"][i]
                    )
                ),
            }
        )

    return rows