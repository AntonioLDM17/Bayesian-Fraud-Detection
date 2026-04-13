from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.load_data import load_data
from src.data.preprocess import get_feature_columns
from src.data.split import split_features_target
from src.models.baseline import build_baseline_models


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate model on a dataset."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, float] = {
        "pr_auc": float(average_precision_score(y, y_proba)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
    }

    return metrics


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_and_save(
    data_path: str | Path,
    output_dir: str | Path = "experiments/baseline_results",
) -> dict[str, Any]:
    """Train baseline models, evaluate them, and save outputs."""
    output_dir = ensure_dir(output_dir)

    df = load_data(data_path, add_row_id=True)
    splits = split_features_target(df, drop_row_id_from_X=True)

    X_train = splits.X_train
    X_val = splits.X_val
    X_test = splits.X_test
    y_train = splits.y_train
    y_val = splits.y_val
    y_test = splits.y_test

    feature_names = get_feature_columns(X_train.columns, exclude_target=False, exclude_row_id=True)
    models = build_baseline_models(feature_names)

    all_results: dict[str, Any] = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)

        print(f"Validation metrics for {model_name}:")
        for metric_name, metric_value in val_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        print(f"Test metrics for {model_name}:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        all_results[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "model_path": str(model_path),
        }

    results_path = output_dir / "metrics.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved metrics to: {results_path}")
    return all_results


if __name__ == "__main__":
    DATA_PATH = "data/raw/creditcard.csv"
    train_and_save(DATA_PATH)