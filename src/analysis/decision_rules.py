from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    BLOCK_PROBABILITY_THRESHOLD,
    BNN_DECISION_DIR,
    DECISION_CSV_PATH,
    DECISION_SUMMARY_PATH,
    UNCERTAINTY_CSV_PATH,
    UNCERTAINTY_THRESHOLD,
)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def assign_decision(
    probability: float,
    uncertainty: float,
    optimal_threshold: float,
    block_probability_threshold: float = BLOCK_PROBABILITY_THRESHOLD,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
) -> str:
    """
    Decision policy using model-specific optimal threshold.

    NEW LOGIC:
    - BLOCK: very high probability + low uncertainty
    - REVIEW: above model threshold OR high uncertainty
    - ACCEPT: below threshold and low uncertainty
    """

    # 🔴 HIGH CONFIDENCE FRAUD
    if probability >= block_probability_threshold and uncertainty <= uncertainty_threshold:
        return "BLOCK"

    # 🟡 BORDERLINE OR UNCERTAIN
    if probability >= optimal_threshold or uncertainty > uncertainty_threshold:
        return "REVIEW"

    # 🟢 SAFE
    return "ACCEPT"


def add_decisions(
    df: pd.DataFrame,
    optimal_threshold: float,
    block_probability_threshold: float = BLOCK_PROBABILITY_THRESHOLD,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
) -> pd.DataFrame:

    df = df.copy()

    df["decision"] = df.apply(
        lambda row: assign_decision(
            probability=row["predicted_probability"],
            uncertainty=row["uncertainty_std"],
            optimal_threshold=optimal_threshold,
            block_probability_threshold=block_probability_threshold,
            uncertainty_threshold=uncertainty_threshold,
        ),
        axis=1,
    )

    return df


def summarize_decisions(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}

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
    uncertainty_csv_path: str | Path = UNCERTAINTY_CSV_PATH,
    evaluation_json_path: str | Path = "experiments/full_evaluation.json",
    output_dir: str | Path = BNN_DECISION_DIR,
) -> dict[str, Any]:

    uncertainty_csv_path = Path(uncertainty_csv_path)
    evaluation_json_path = Path(evaluation_json_path)

    if not uncertainty_csv_path.exists():
        raise FileNotFoundError(f"Missing uncertainty file: {uncertainty_csv_path}")

    if not evaluation_json_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {evaluation_json_path}")

    output_dir = ensure_dir(output_dir)

    df = pd.read_csv(uncertainty_csv_path)

    # 🔥 EXTRA: leer threshold óptimo del BNN
    with evaluation_json_path.open("r", encoding="utf-8") as f:
        eval_results = json.load(f)

    bnn_key = [k for k in eval_results["models"] if "bnn" in k][0]
    optimal_threshold = eval_results["models"][bnn_key]["selected_threshold_from_validation"]

    print(f"Using optimal threshold from validation: {optimal_threshold:.4f}")

    df_decisions = add_decisions(
        df,
        optimal_threshold=optimal_threshold,
    )

    summary = summarize_decisions(df_decisions)
    summary["optimal_threshold"] = float(optimal_threshold)

    csv_out = output_dir / Path(DECISION_CSV_PATH).name
    json_out = output_dir / Path(DECISION_SUMMARY_PATH).name

    df_decisions.to_csv(csv_out, index=False)

    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved decisions CSV to: {csv_out}")
    print(f"Saved decision summary JSON to: {json_out}")

    return summary


if __name__ == "__main__":
    run_decision_analysis()