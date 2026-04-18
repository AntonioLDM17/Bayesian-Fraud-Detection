from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BLOCK_PROBABILITY_THRESHOLD,
    BNN_CHECKPOINT_PATH,
    BNN_PREPROCESSOR_PATH,
    DATA_PATH,
    FULL_EVALUATION_PATH,
    UNCERTAINTY_THRESHOLD,
)
from src.data.load_data import load_data
from src.data.split import split_features_target
from src.inference.bnn_inference import (
    load_bnn_artifacts_for_inference,
    predict_proba_and_uncertainty,
    transform_features,
)


OUTPUT_DIR = Path("experiments") / "decision_rates_by_model"
OUTPUT_JSON = OUTPUT_DIR / "decision_rates_by_model.json"
OUTPUT_CSV = OUTPUT_DIR / "decision_rates_by_model.csv"


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_full_evaluation(path: str | Path = FULL_EVALUATION_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_test_split():
    df = load_data(DATA_PATH, add_row_id=True)
    splits = split_features_target(df, drop_row_id_from_X=True)
    return splits.X_test.copy(), splits.y_test.to_numpy(dtype=int)


# ============================================================
# Decision rules
# ============================================================

def assign_decision_deterministic(p: float, thr: float) -> str:
    if p >= BLOCK_PROBABILITY_THRESHOLD:
        return "BLOCK"
    if p >= thr:
        return "REVIEW"
    return "ACCEPT"


def assign_decision_bnn(p: float, u: float, thr: float) -> str:
    if p >= BLOCK_PROBABILITY_THRESHOLD and u <= UNCERTAINTY_THRESHOLD:
        return "BLOCK"
    if p >= thr or u > UNCERTAINTY_THRESHOLD:
        return "REVIEW"
    return "ACCEPT"


# ============================================================
# Metrics
# ============================================================

def compute_decision_stats(
    decisions: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float | int | None]:

    total = len(decisions)

    def _mask(label):
        return decisions == label

    def _rate(label):
        return float((_mask(label)).sum() / total)

    def _fraud_rate(label):
        mask = _mask(label)
        if mask.sum() == 0:
            return None
        return float(y_true[mask].mean())

    return {
        "num_samples": total,

        "accept_rate": _rate("ACCEPT"),
        "review_rate": _rate("REVIEW"),
        "block_rate": _rate("BLOCK"),

        "accept_fraud_rate": _fraud_rate("ACCEPT"),
        "review_fraud_rate": _fraud_rate("REVIEW"),
        "block_fraud_rate": _fraud_rate("BLOCK"),
    }


# ============================================================
# Main
# ============================================================

def run() -> dict[str, Any]:
    ensure_dir(OUTPUT_DIR)

    full_eval = load_full_evaluation()
    X_test, y_true = get_test_split()

    results: dict[str, Any] = {"models": {}}
    rows = []

    # --------------------------
    # Deterministic models
    # --------------------------
    for model_name, model_info in full_eval["models"].items():
        if model_info["artifact_type"] == "bnn":
            continue

        model = joblib.load(model_info["model_path"])
        thr = float(model_info["selected_threshold_from_validation"])

        probs = model.predict_proba(X_test)[:, 1]

        decisions = np.array([
            assign_decision_deterministic(p, thr) for p in probs
        ])

        stats = compute_decision_stats(decisions, y_true)

        results["models"][model_name] = stats

        rows.append({
            "model": model_name,

            "accept_%": 100 * stats["accept_rate"],
            "review_%": 100 * stats["review_rate"],
            "block_%": 100 * stats["block_rate"],

            "fraud_in_accept_%": None if stats["accept_fraud_rate"] is None else 100 * stats["accept_fraud_rate"],
            "fraud_in_review_%": None if stats["review_fraud_rate"] is None else 100 * stats["review_fraud_rate"],
            "fraud_in_block_%": None if stats["block_fraud_rate"] is None else 100 * stats["block_fraud_rate"],
        })

    # --------------------------
    # BNN
    # --------------------------
    bnn_artifacts = load_bnn_artifacts_for_inference(
        checkpoint_path=BNN_CHECKPOINT_PATH,
        preprocessor_path=BNN_PREPROCESSOR_PATH,
    )

    for model_name, model_info in full_eval["models"].items():
        if model_info["artifact_type"] != "bnn":
            continue

        thr = float(model_info["selected_threshold_from_validation"])

        X_trans = transform_features(
            X_test,
            bnn_artifacts["preprocessor"],
            bnn_artifacts["feature_names"],
        )

        probs, uncert = predict_proba_and_uncertainty(
            model=bnn_artifacts["model"],
            guide=bnn_artifacts["guide"],
            X_transformed=X_trans,
            num_mc_samples=200,
        )

        decisions = np.array([
            assign_decision_bnn(p, u, thr)
            for p, u in zip(probs, uncert)
        ])

        stats = compute_decision_stats(decisions, y_true)

        results["models"][model_name] = stats

        rows.append({
            "model": model_name,

            "accept_%": 100 * stats["accept_rate"],
            "review_%": 100 * stats["review_rate"],
            "block_%": 100 * stats["block_rate"],

            "fraud_in_accept_%": None if stats["accept_fraud_rate"] is None else 100 * stats["accept_fraud_rate"],
            "fraud_in_review_%": None if stats["review_fraud_rate"] is None else 100 * stats["review_fraud_rate"],
            "fraud_in_block_%": None if stats["block_fraud_rate"] is None else 100 * stats["block_fraud_rate"],
        })

    # --------------------------
    # Save
    # --------------------------
    df = pd.DataFrame(rows).sort_values("model")

    df.to_csv(OUTPUT_CSV, index=False)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Decision breakdown by model ===\n")
    print(df.to_string(index=False))

    print(f"\nSaved CSV to: {OUTPUT_CSV}")
    print(f"Saved JSON to: {OUTPUT_JSON}")

    return results


if __name__ == "__main__":
    run()