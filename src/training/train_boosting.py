from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models.boosting import build_boosting_models


RANDOM_STATE = 42
TARGET_COLUMN = "Class"


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load fraud dataset from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
):
    """
    Split data into train/validation/test using stratification.

    Must remain consistent with baseline training.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    adjusted_val_size = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=adjusted_val_size,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Compute scale_pos_weight = negative / positive
    for imbalanced binary classification.
    """
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0:
        raise ValueError("No positive samples found in training set.")

    return negative_count / positive_count


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate model on a dataset."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
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
    output_dir: str | Path = "experiments/boosting_results",
) -> dict[str, Any]:
    """
    Train boosting models, evaluate them, and save outputs.
    """
    output_dir = ensure_dir(output_dir)

    df = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    feature_names = list(X_train.columns)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

    models = build_boosting_models(
        feature_names=feature_names,
        scale_pos_weight=scale_pos_weight,
    )

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
            "scale_pos_weight": float(scale_pos_weight),
        }

    results_path = output_dir / "metrics.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved metrics to: {results_path}")
    return all_results


if __name__ == "__main__":
    DATA_PATH = "data/raw/creditcard.csv"
    train_and_save(DATA_PATH)