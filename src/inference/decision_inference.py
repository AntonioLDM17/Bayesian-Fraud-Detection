from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.decision_rules import assign_decision
from src.config import (
    BLOCK_PROBABILITY_THRESHOLD,
    FULL_EVALUATION_PATH,
    UNCERTAINTY_THRESHOLD,
)
from src.inference.bnn_inference import predict_batch, predict_single


def load_decision_context(
    evaluation_json_path: str | Path = FULL_EVALUATION_PATH,
    block_probability_threshold: float = BLOCK_PROBABILITY_THRESHOLD,
    uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
) -> dict[str, float]:
    """
    Load decision thresholds needed at inference time.

    Returns:
    - optimal_threshold
    - block_probability_threshold
    - uncertainty_threshold
    """
    evaluation_json_path = Path(evaluation_json_path)
    if not evaluation_json_path.exists():
        raise FileNotFoundError(
            f"Evaluation JSON not found: {evaluation_json_path}"
        )

    with evaluation_json_path.open("r", encoding="utf-8") as f:
        eval_results = json.load(f)

    model_keys = list(eval_results.get("models", {}).keys())
    bnn_keys = [k for k in model_keys if "bnn" in k.lower()]
    if not bnn_keys:
        raise ValueError(
            "Could not find a BNN entry in the evaluation JSON."
        )

    bnn_key = bnn_keys[0]
    optimal_threshold = eval_results["models"][bnn_key]["selected_threshold_from_validation"]

    return {
        "optimal_threshold": float(optimal_threshold),
        "block_probability_threshold": float(block_probability_threshold),
        "uncertainty_threshold": float(uncertainty_threshold),
    }


def predict_decision_for_batch(
    X: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    evaluation_json_path: str | Path = FULL_EVALUATION_PATH,
    num_mc_samples: int = 200,
    include_input_columns: bool = False,
) -> pd.DataFrame:
    """
    Predict BNN probability, uncertainty, and final decision for one or many samples.

    Returns a DataFrame with:
    - predicted_probability
    - uncertainty_std
    - decision
    - optimal_threshold
    - block_probability_threshold
    - uncertainty_threshold
    """
    context = load_decision_context(evaluation_json_path=evaluation_json_path)

    preds = predict_batch(
        X=X,
        num_mc_samples=num_mc_samples,
        include_input_columns=include_input_columns,
    ).copy()

    preds["decision"] = preds.apply(
        lambda row: assign_decision(
            probability=float(row["predicted_probability"]),
            uncertainty=float(row["uncertainty_std"]),
            optimal_threshold=context["optimal_threshold"],
            block_probability_threshold=context["block_probability_threshold"],
            uncertainty_threshold=context["uncertainty_threshold"],
        ),
        axis=1,
    )

    preds["optimal_threshold"] = context["optimal_threshold"]
    preds["block_probability_threshold"] = context["block_probability_threshold"]
    preds["uncertainty_threshold"] = context["uncertainty_threshold"]

    return preds


def predict_decision_for_single_transaction(
    x: dict[str, Any] | pd.Series,
    evaluation_json_path: str | Path = FULL_EVALUATION_PATH,
    num_mc_samples: int = 200,
) -> dict[str, float | str]:
    """
    Predict final decision for a single transaction.

    Returns:
    - predicted_probability
    - uncertainty_std
    - decision
    - optimal_threshold
    - block_probability_threshold
    - uncertainty_threshold
    """
    context = load_decision_context(evaluation_json_path=evaluation_json_path)
    pred = predict_single(
        x=x,
        num_mc_samples=num_mc_samples,
    )

    decision = assign_decision(
        probability=pred["predicted_probability"],
        uncertainty=pred["uncertainty_std"],
        optimal_threshold=context["optimal_threshold"],
        block_probability_threshold=context["block_probability_threshold"],
        uncertainty_threshold=context["uncertainty_threshold"],
    )

    return {
        "predicted_probability": float(pred["predicted_probability"]),
        "uncertainty_std": float(pred["uncertainty_std"]),
        "decision": decision,
        "optimal_threshold": float(context["optimal_threshold"]),
        "block_probability_threshold": float(context["block_probability_threshold"]),
        "uncertainty_threshold": float(context["uncertainty_threshold"]),
    }