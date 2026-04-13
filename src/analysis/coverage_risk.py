from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BNN_UNCERTAINTY_DIR,
    PLOT_DPI,
    UNCERTAINTY_CSV_PATH,
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_coverage_risk(
    df: pd.DataFrame,
    uncertainty_col: str = "uncertainty_std",
    num_thresholds: int = 50,
) -> list[dict[str, float]]:
    """
    Compute coverage-risk curve.

    Coverage = fraction of samples retained after rejecting high-uncertainty cases.
    Risk = error rate on retained samples.
    """
    max_uncertainty = float(df[uncertainty_col].max())

    if max_uncertainty == 0.0:
        thresholds = np.array([0.0], dtype=float)
    else:
        thresholds = np.linspace(0.0, max_uncertainty, num_thresholds)

    results: list[dict[str, float]] = []

    for thr in thresholds:
        subset = df[df[uncertainty_col] <= thr]

        if len(subset) == 0:
            continue

        coverage = len(subset) / len(df)
        risk = float((subset["y_pred"] != subset["y_true"]).mean())

        results.append(
            {
                "uncertainty_threshold": float(thr),
                "coverage": float(coverage),
                "risk": risk,
            }
        )

    return results


def compute_selective_prediction(
    df: pd.DataFrame,
    uncertainty_col: str = "uncertainty_std",
    num_thresholds: int = 20,
) -> dict[str, dict[str, float | int | None]]:
    """
    Compute simple selective prediction metrics across uncertainty thresholds.

    For each threshold:
    - retain only low-uncertainty samples
    - report coverage and accuracy
    """
    max_uncertainty = float(df[uncertainty_col].max())

    if max_uncertainty == 0.0:
        thresholds = np.array([0.0], dtype=float)
    else:
        thresholds = np.linspace(0.0, max_uncertainty, num_thresholds)

    results: dict[str, dict[str, float | int | None]] = {}

    for thr in thresholds:
        subset = df[df[uncertainty_col] <= thr]

        if len(subset) == 0:
            results[f"{float(thr):.8f}"] = {
                "coverage": 0.0,
                "accuracy": None,
                "num_samples": 0,
            }
            continue

        coverage = len(subset) / len(df)
        accuracy = float((subset["y_pred"] == subset["y_true"]).mean())

        results[f"{float(thr):.8f}"] = {
            "coverage": float(coverage),
            "accuracy": accuracy,
            "num_samples": int(len(subset)),
        }

    return results


def summarize_coverage_risk(
    coverage_risk_results: list[dict[str, float]],
    selective_results: dict[str, dict[str, float | int | None]],
) -> dict[str, Any]:
    """
    Build a compact summary with a few useful operating points.
    """
    summary: dict[str, Any] = {
        "coverage_risk_points": len(coverage_risk_results),
        "selective_prediction_points": len(selective_results),
    }

    if coverage_risk_results:
        min_risk_point = min(coverage_risk_results, key=lambda x: x["risk"])
        max_coverage_point = max(coverage_risk_results, key=lambda x: x["coverage"])

        summary["best_low_risk_point"] = min_risk_point
        summary["max_coverage_point"] = max_coverage_point

        # nearest practical points
        target_coverages = [0.5, 0.7, 0.9]
        summary["nearest_target_coverages"] = {}

        for target in target_coverages:
            nearest = min(
                coverage_risk_results,
                key=lambda x: abs(x["coverage"] - target),
            )
            summary["nearest_target_coverages"][str(target)] = nearest

    return summary


def plot_coverage_risk(
    results: list[dict[str, float]],
    output_path: Path,
) -> None:
    """
    Plot coverage vs risk.
    """
    if not results:
        return

    coverage = [r["coverage"] for r in results]
    risk = [r["risk"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(coverage, risk, marker="o", markersize=3)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (error rate)")
    plt.title("Coverage vs Risk")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def plot_coverage_accuracy(
    selective_results: dict[str, dict[str, float | int | None]],
    output_path: Path,
) -> None:
    """
    Plot coverage vs accuracy for selective prediction.
    """
    if not selective_results:
        return

    coverage = []
    accuracy = []

    for metrics in selective_results.values():
        acc = metrics["accuracy"]
        if acc is None:
            continue
        coverage.append(float(metrics["coverage"]))
        accuracy.append(float(acc))

    if not coverage:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(coverage, accuracy, marker="o", markersize=3)
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Coverage vs Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def run_coverage_risk_analysis(
    uncertainty_csv_path: str | Path = UNCERTAINTY_CSV_PATH,
    output_dir: str | Path = BNN_UNCERTAINTY_DIR,
) -> dict[str, Any]:
    """
    Run coverage-risk and selective prediction analysis from the
    uncertainty per-sample CSV generated by uncertainty.py.
    """
    uncertainty_csv_path = Path(uncertainty_csv_path)
    if not uncertainty_csv_path.exists():
        raise FileNotFoundError(f"Could not find: {uncertainty_csv_path}")

    output_dir = ensure_dir(output_dir)

    df = pd.read_csv(uncertainty_csv_path)

    required_cols = {
        "y_true",
        "y_pred",
        "predicted_probability",
        "uncertainty_std",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    coverage_risk_results = compute_coverage_risk(df)
    selective_results = compute_selective_prediction(df)
    summary = summarize_coverage_risk(coverage_risk_results, selective_results)

    coverage_risk_json = output_dir / "test_coverage_risk.json"
    selective_json = output_dir / "test_selective_prediction.json"
    summary_json = output_dir / "test_coverage_risk_summary.json"
    coverage_risk_plot = output_dir / "test_coverage_vs_risk.png"
    coverage_accuracy_plot = output_dir / "test_coverage_vs_accuracy.png"

    with coverage_risk_json.open("w", encoding="utf-8") as f:
        json.dump(coverage_risk_results, f, indent=2)

    with selective_json.open("w", encoding="utf-8") as f:
        json.dump(selective_results, f, indent=2)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_coverage_risk(coverage_risk_results, coverage_risk_plot)
    plot_coverage_accuracy(selective_results, coverage_accuracy_plot)

    print(f"Saved coverage-risk JSON to: {coverage_risk_json}")
    print(f"Saved selective prediction JSON to: {selective_json}")
    print(f"Saved coverage-risk summary JSON to: {summary_json}")
    print(f"Saved coverage-vs-risk plot to: {coverage_risk_plot}")
    print(f"Saved coverage-vs-accuracy plot to: {coverage_accuracy_plot}")

    return summary


if __name__ == "__main__":
    run_coverage_risk_analysis()