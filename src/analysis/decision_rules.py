from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def assign_decision(
    probability: float,
    uncertainty: float,
    block_probability_threshold: float = 0.80,
    review_probability_threshold: float = 0.30,
    uncertainty_threshold: float = 0.10,
) -> str:
    """
    Decision policy using both predicted fraud probability and uncertainty.

    Rules:
    - BLOCK: high probability + low uncertainty
    - REVIEW: medium/high probability or high uncertainty
    - ACCEPT: low probability
    """
    if probability >= block_probability_threshold and uncertainty <= uncertainty_threshold:
        return "BLOCK"

    if probability >= 0.50 and uncertainty > uncertainty_threshold:
        return "REVIEW"

    if probability >= review_probability_threshold:
        return "REVIEW"

    return "ACCEPT"


def add_decisions(
    df: pd.DataFrame,
    block_probability_threshold: float = 0.80,
    review_probability_threshold: float = 0.30,
    uncertainty_threshold: float = 0.10,
) -> pd.DataFrame:
    """Add decision column to per-sample uncertainty dataframe."""
    df = df.copy()

    df["decision"] = df.apply(
        lambda row: assign_decision(
            probability=row["predicted_probability"],
            uncertainty=row["uncertainty_std"],
            block_probability_threshold=block_probability_threshold,
            review_probability_threshold=review_probability_threshold,
            uncertainty_threshold=uncertainty_threshold,
        ),
        axis=1,
    )

    return df


def summarize_decisions(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize decision distribution and fraud composition."""
    summary: dict[str, Any] = {}

    summary["row_id_included"] = "row_id" in df.columns

    decision_counts = df["decision"].value_counts(dropna=False).to_dict()
    summary["decision_counts"] = {str(k): int(v) for k, v in decision_counts.items()}

    summary["by_decision"] = {}
    for decision in ["ACCEPT", "REVIEW", "BLOCK"]:
        sub = df[df["decision"] == decision]

        if len(sub) == 0:
            summary["by_decision"][decision] = {
                "count": 0,
                "fraud_rate": None,
                "mean_probability": None,
                "mean_uncertainty": None,
            }
            continue

        summary["by_decision"][decision] = {
            "count": int(len(sub)),
            "fraud_rate": float(sub["y_true"].mean()),
            "mean_probability": float(sub["predicted_probability"].mean()),
            "mean_uncertainty": float(sub["uncertainty_std"].mean()),
        }

    return summary


def run_decision_analysis(
    uncertainty_csv_path: str | Path = "experiments/bnn_results/uncertainty_analysis/test_uncertainty_per_sample.csv",
    output_dir: str | Path = "experiments/bnn_results/decision_analysis",
    block_probability_threshold: float = 0.80,
    review_probability_threshold: float = 0.30,
    uncertainty_threshold: float = 0.10,
) -> dict[str, Any]:
    """
    Apply uncertainty-aware decision policy to per-sample BNN outputs.
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
        "confusion_type",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # row_id is optional for backward compatibility, but should be present now.
    row_id_present = "row_id" in df.columns

    df_decisions = add_decisions(
        df,
        block_probability_threshold=block_probability_threshold,
        review_probability_threshold=review_probability_threshold,
        uncertainty_threshold=uncertainty_threshold,
    )

    summary = summarize_decisions(df_decisions)
    summary["policy"] = {
        "block_probability_threshold": block_probability_threshold,
        "review_probability_threshold": review_probability_threshold,
        "uncertainty_threshold": uncertainty_threshold,
    }
    summary["row_id_included"] = row_id_present

    csv_out = output_dir / "test_decisions_per_sample.csv"
    json_out = output_dir / "test_decision_summary.json"

    df_decisions.to_csv(csv_out, index=False)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved decisions CSV to: {csv_out}")
    print(f"Saved decision summary JSON to: {json_out}")

    return summary


if __name__ == "__main__":
    run_decision_analysis()