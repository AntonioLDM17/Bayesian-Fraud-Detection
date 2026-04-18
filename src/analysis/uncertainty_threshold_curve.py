from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BLOCK_PROBABILITY_THRESHOLD,
    FULL_EVALUATION_PATH,
    PLOT_DPI,
    UNCERTAINTY_THRESHOLD,
)


UNCERTAINTY_CSV_PATH = Path(
    "experiments/bnn_results/uncertainty_analysis/test_uncertainty_per_sample.csv"
)
OUTPUT_DIR = Path("experiments/bnn_results/uncertainty_analysis")
OUTPUT_CSV_PATH = OUTPUT_DIR / "uncertainty_threshold_curve.csv"
OUTPUT_JSON_PATH = OUTPUT_DIR / "uncertainty_threshold_curve_summary.json"
OUTPUT_PLOT_DECISIONS = OUTPUT_DIR / "uncertainty_threshold_vs_decision_rates.png"
OUTPUT_PLOT_FRAUD = OUTPUT_DIR / "uncertainty_threshold_vs_fraud_rates.png"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_optimal_threshold(
    evaluation_json_path: str | Path = FULL_EVALUATION_PATH,
) -> float:
    with Path(evaluation_json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    for model_name, model_info in data["models"].items():
        if model_info.get("artifact_type") == "bnn":
            return float(model_info["selected_threshold_from_validation"])

    raise ValueError("Could not find BNN threshold in full_evaluation.json")


def assign_decision(
    probability: float,
    uncertainty: float,
    optimal_threshold: float,
    uncertainty_threshold: float,
    block_probability_threshold: float = BLOCK_PROBABILITY_THRESHOLD,
) -> str:
    if probability >= block_probability_threshold and uncertainty <= uncertainty_threshold:
        return "BLOCK"
    if probability >= optimal_threshold or uncertainty > uncertainty_threshold:
        return "REVIEW"
    return "ACCEPT"


def safe_rate(mask: np.ndarray, y_true: np.ndarray) -> float | None:
    if mask.sum() == 0:
        return None
    return float(y_true[mask].mean())


def compute_stats_for_threshold(
    df: pd.DataFrame,
    optimal_threshold: float,
    uncertainty_threshold: float,
) -> dict[str, Any]:
    probabilities = df["predicted_probability"].to_numpy(dtype=float)
    uncertainties = df["uncertainty_std"].to_numpy(dtype=float)
    y_true = df["y_true"].to_numpy(dtype=int)

    decisions = np.array(
        [
            assign_decision(
                probability=p,
                uncertainty=u,
                optimal_threshold=optimal_threshold,
                uncertainty_threshold=uncertainty_threshold,
            )
            for p, u in zip(probabilities, uncertainties)
        ],
        dtype=object,
    )

    accept_mask = decisions == "ACCEPT"
    review_mask = decisions == "REVIEW"
    block_mask = decisions == "BLOCK"

    total = len(decisions)

    return {
        "uncertainty_threshold": float(uncertainty_threshold),
        "accept_rate": float(accept_mask.sum() / total),
        "review_rate": float(review_mask.sum() / total),
        "block_rate": float(block_mask.sum() / total),
        "fraud_in_accept_rate": safe_rate(accept_mask, y_true),
        "fraud_in_review_rate": safe_rate(review_mask, y_true),
        "fraud_in_block_rate": safe_rate(block_mask, y_true),
        "accept_count": int(accept_mask.sum()),
        "review_count": int(review_mask.sum()),
        "block_count": int(block_mask.sum()),
    }


def choose_candidate_thresholds(df: pd.DataFrame, n_grid: int = 50) -> np.ndarray:
    max_unc = float(df["uncertainty_std"].max())
    if max_unc <= 0:
        return np.array([0.0])

    # grid uniforme + percentiles para tener puntos interpretables
    grid = np.linspace(0.0, max_unc, n_grid)
    percentiles = np.percentile(
        df["uncertainty_std"].to_numpy(dtype=float),
        [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99],
    )
    values = np.unique(np.concatenate([grid, percentiles, np.array([UNCERTAINTY_THRESHOLD])]))
    return np.sort(values)


def plot_decision_rates(results_df: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["uncertainty_threshold"], results_df["accept_rate"], label="ACCEPT")
    plt.plot(results_df["uncertainty_threshold"], results_df["review_rate"], label="REVIEW")
    plt.plot(results_df["uncertainty_threshold"], results_df["block_rate"], label="BLOCK")
    plt.xlabel("Uncertainty threshold")
    plt.ylabel("Fraction of samples")
    plt.title("Decision rates vs uncertainty threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def plot_fraud_rates(results_df: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))

    for col, label in [
        ("fraud_in_accept_rate", "Fraud in ACCEPT"),
        ("fraud_in_review_rate", "Fraud in REVIEW"),
        ("fraud_in_block_rate", "Fraud in BLOCK"),
    ]:
        series = results_df[col].astype(float)
        plt.plot(results_df["uncertainty_threshold"], series, label=label)

    plt.xlabel("Uncertainty threshold")
    plt.ylabel("Fraud rate inside decision bucket")
    plt.title("Fraud concentration vs uncertainty threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def run() -> dict[str, Any]:
    ensure_dir(OUTPUT_DIR)

    if not UNCERTAINTY_CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {UNCERTAINTY_CSV_PATH}")

    df = pd.read_csv(UNCERTAINTY_CSV_PATH)
    optimal_threshold = load_optimal_threshold(FULL_EVALUATION_PATH)

    candidate_thresholds = choose_candidate_thresholds(df, n_grid=60)

    rows = [
        compute_stats_for_threshold(
            df=df,
            optimal_threshold=optimal_threshold,
            uncertainty_threshold=float(t),
        )
        for t in candidate_thresholds
    ]

    results_df = pd.DataFrame(rows).sort_values("uncertainty_threshold").reset_index(drop=True)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)

    plot_decision_rates(results_df, OUTPUT_PLOT_DECISIONS)
    plot_fraud_rates(results_df, OUTPUT_PLOT_FRAUD)

    # algunos puntos útiles para el informe
    current_row = results_df.iloc[
        (results_df["uncertainty_threshold"] - UNCERTAINTY_THRESHOLD).abs().argmin()
    ]

    # threshold "útil" de ejemplo: mínimo fraud_in_accept_rate con accept_rate >= 0.5
    candidate_df = results_df[
        (results_df["accept_rate"] >= 0.50)
        & results_df["fraud_in_accept_rate"].notna()
    ].copy()

    suggested = None
    if not candidate_df.empty:
        suggested_row = candidate_df.sort_values(
            ["fraud_in_accept_rate", "review_rate"],
            ascending=[True, True],
        ).iloc[0]
        suggested = {
            "uncertainty_threshold": float(suggested_row["uncertainty_threshold"]),
            "accept_rate": float(suggested_row["accept_rate"]),
            "review_rate": float(suggested_row["review_rate"]),
            "block_rate": float(suggested_row["block_rate"]),
            "fraud_in_accept_rate": float(suggested_row["fraud_in_accept_rate"]),
            "fraud_in_review_rate": (
                None if pd.isna(suggested_row["fraud_in_review_rate"])
                else float(suggested_row["fraud_in_review_rate"])
            ),
            "fraud_in_block_rate": (
                None if pd.isna(suggested_row["fraud_in_block_rate"])
                else float(suggested_row["fraud_in_block_rate"])
            ),
            "selection_rule": "minimum fraud_in_accept_rate subject to accept_rate >= 0.50",
        }

    summary = {
        "metadata": {
            "uncertainty_csv_path": str(UNCERTAINTY_CSV_PATH),
            "evaluation_json_path": str(FULL_EVALUATION_PATH),
            "optimal_probability_threshold": float(optimal_threshold),
            "block_probability_threshold": float(BLOCK_PROBABILITY_THRESHOLD),
            "current_uncertainty_threshold_in_config": float(UNCERTAINTY_THRESHOLD),
        },
        "current_threshold_operating_point": {
            "uncertainty_threshold": float(current_row["uncertainty_threshold"]),
            "accept_rate": float(current_row["accept_rate"]),
            "review_rate": float(current_row["review_rate"]),
            "block_rate": float(current_row["block_rate"]),
            "fraud_in_accept_rate": (
                None if pd.isna(current_row["fraud_in_accept_rate"])
                else float(current_row["fraud_in_accept_rate"])
            ),
            "fraud_in_review_rate": (
                None if pd.isna(current_row["fraud_in_review_rate"])
                else float(current_row["fraud_in_review_rate"])
            ),
            "fraud_in_block_rate": (
                None if pd.isna(current_row["fraud_in_block_rate"])
                else float(current_row["fraud_in_block_rate"])
            ),
        },
        "suggested_threshold": suggested,
    }

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved curve CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved summary JSON to: {OUTPUT_JSON_PATH}")
    print(f"Saved decision plot to: {OUTPUT_PLOT_DECISIONS}")
    print(f"Saved fraud plot to: {OUTPUT_PLOT_FRAUD}")

    print("\nCurrent operating point:")
    print(current_row.to_string())

    if suggested is not None:
        print("\nSuggested threshold:")
        for k, v in suggested.items():
            print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    run()