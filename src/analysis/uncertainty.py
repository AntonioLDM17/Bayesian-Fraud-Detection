from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import Predictive

from src.config import (
    BNN_CHECKPOINT_PATH,
    BNN_PREPROCESSOR_PATH,
    BNN_UNCERTAINTY_DIR,
    DATA_PATH,
    DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
    DEFAULT_THRESHOLD,
    DEVICE,
    PLOT_DPI,
)
from src.data.load_data import ROW_ID_COLUMN, load_data
from src.data.preprocess import get_feature_columns
from src.data.split import split_features_target
from src.evaluation.thresholds import find_best_f1_threshold
from src.models.bnn import BayesianMLP


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_bnn_from_checkpoint(checkpoint: dict[str, Any]) -> BayesianMLP:
    """
    Supports both the new and old checkpoint formats.
    """
    input_dim = checkpoint["input_dim"]
    prior_scale = checkpoint["prior_scale"]

    if "hidden_dim_1" in checkpoint and "hidden_dim_2" in checkpoint:
        return BayesianMLP(
            input_dim=input_dim,
            hidden_dim_1=checkpoint["hidden_dim_1"],
            hidden_dim_2=checkpoint["hidden_dim_2"],
            prior_scale=prior_scale,
            dropout_rate=checkpoint.get("dropout_rate", 0.1),
        )

    return BayesianMLP(
        input_dim=input_dim,
        hidden_dim_1=checkpoint["hidden_dim"],
        hidden_dim_2=max(checkpoint["hidden_dim"] // 2, 16),
        prior_scale=prior_scale,
        dropout_rate=0.0,
    )


def load_bnn_artifacts(
    checkpoint_path: str | Path,
    preprocessor_path: str | Path,
) -> tuple[BayesianMLP, Any, Any, list[str] | None]:
    """
    Load BNN checkpoint and preprocessor.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    preprocessor = joblib.load(preprocessor_path)

    model = _build_bnn_from_checkpoint(checkpoint).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pyro.clear_param_store()
    pyro.get_param_store().set_state(checkpoint["pyro_param_store"])

    blocked_model = pyro.poutine.block(model, hide=["obs"])
    guide = pyro.infer.autoguide.AutoDiagonalNormal(blocked_model)

    feature_names = checkpoint.get("feature_names")

    return model, guide, preprocessor, feature_names


@torch.no_grad()
def predict_with_uncertainty(
    model: BayesianMLP,
    guide,
    x: torch.Tensor,
    num_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
    - predictive mean probability
    - predictive std of probability across MC samples
    """
    model.eval()
    x = x.to(DEVICE)

    all_mean_probs = []
    all_std_probs = []

    for start_idx in range(0, x.shape[0], batch_size):
        batch_x = x[start_idx : start_idx + batch_size]

        predictive = Predictive(
            model=model,
            guide=guide,
            num_samples=num_samples,
            return_sites=["_RETURN"],
        )
        samples = predictive(batch_x)
        logits_samples = samples["_RETURN"]
        prob_samples = torch.sigmoid(logits_samples)

        mean_probs = prob_samples.mean(dim=0)
        std_probs = prob_samples.std(dim=0)

        all_mean_probs.append(mean_probs.cpu())
        all_std_probs.append(std_probs.cpu())

    return (
        torch.cat(all_mean_probs, dim=0).numpy(),
        torch.cat(all_std_probs, dim=0).numpy(),
    )


def assign_confusion_label(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Assign TP / FP / TN / FN per sample."""
    labels = []

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            labels.append("TP")
        elif yt == 0 and yp == 1:
            labels.append("FP")
        elif yt == 0 and yp == 0:
            labels.append("TN")
        elif yt == 1 and yp == 0:
            labels.append("FN")
        else:
            labels.append("UNKNOWN")

    return np.array(labels, dtype=object)


def compute_coverage_risk(
    df: pd.DataFrame,
    uncertainty_col: str = "uncertainty_std",
    num_thresholds: int = 50,
) -> list[dict[str, float]]:
    """
    Compute coverage-risk tradeoff as uncertainty threshold increases.

    Coverage = fraction of kept samples
    Risk = error rate on kept samples
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


def selective_metrics(
    df: pd.DataFrame,
    uncertainty_threshold: float,
) -> dict[str, float | int | None]:
    """
    Evaluate performance on samples whose uncertainty is below a threshold.
    """
    subset = df[df["uncertainty_std"] <= uncertainty_threshold]

    if len(subset) == 0:
        return {
            "coverage": 0.0,
            "accuracy": None,
            "num_samples": 0,
        }

    coverage = len(subset) / len(df)
    accuracy = float((subset["y_pred"] == subset["y_true"]).mean())

    return {
        "coverage": float(coverage),
        "accuracy": accuracy,
        "num_samples": int(len(subset)),
    }


def summarize_uncertainty(df: pd.DataFrame) -> dict[str, Any]:
    """Build summary statistics for uncertainty analysis."""
    summary: dict[str, Any] = {}

    summary["overall"] = {
        "num_samples": int(len(df)),
        "mean_probability": float(df["predicted_probability"].mean()),
        "mean_uncertainty": float(df["uncertainty_std"].mean()),
        "median_uncertainty": float(df["uncertainty_std"].median()),
    }

    summary["by_true_class"] = {}
    for class_value in sorted(df["y_true"].unique()):
        sub = df[df["y_true"] == class_value]
        summary["by_true_class"][str(class_value)] = {
            "count": int(len(sub)),
            "mean_probability": float(sub["predicted_probability"].mean()),
            "mean_uncertainty": float(sub["uncertainty_std"].mean()),
            "median_uncertainty": float(sub["uncertainty_std"].median()),
        }

    summary["by_confusion_type"] = {}
    for label in ["TP", "FP", "TN", "FN"]:
        sub = df[df["confusion_type"] == label]
        if len(sub) == 0:
            summary["by_confusion_type"][label] = {
                "count": 0,
                "mean_probability": None,
                "mean_uncertainty": None,
                "median_uncertainty": None,
            }
        else:
            summary["by_confusion_type"][label] = {
                "count": int(len(sub)),
                "mean_probability": float(sub["predicted_probability"].mean()),
                "mean_uncertainty": float(sub["uncertainty_std"].mean()),
                "median_uncertainty": float(sub["uncertainty_std"].median()),
            }

    return summary


def plot_uncertainty_by_true_class(df: pd.DataFrame, output_path: Path) -> None:
    """Histogram of uncertainty by true class."""
    plt.figure(figsize=(8, 5))

    for class_value in sorted(df["y_true"].unique()):
        sub = df[df["y_true"] == class_value]
        plt.hist(
            sub["uncertainty_std"],
            bins=40,
            alpha=0.5,
            density=False,
            label=f"True class = {class_value}",
        )

    plt.xlabel("Predictive uncertainty (std of MC probabilities)")
    plt.ylabel("Count")
    plt.title("Uncertainty distribution by true class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def plot_uncertainty_by_confusion_type(df: pd.DataFrame, output_path: Path) -> None:
    """Boxplot of uncertainty by TP / FP / TN / FN."""
    labels = ["TP", "FP", "TN", "FN"]
    data = [df[df["confusion_type"] == label]["uncertainty_std"].values for label in labels]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Predictive uncertainty (std of MC probabilities)")
    plt.title("Uncertainty by confusion type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def plot_probability_vs_uncertainty(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot of predicted probability vs uncertainty."""
    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["predicted_probability"],
        df["uncertainty_std"],
        s=8,
        alpha=0.35,
    )
    plt.xlabel("Predicted fraud probability")
    plt.ylabel("Predictive uncertainty (std)")
    plt.title("Predicted probability vs uncertainty")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def plot_coverage_risk(results: list[dict[str, float]], output_path: Path) -> None:
    """Plot coverage-risk curve."""
    coverage = [r["coverage"] for r in results]
    risk = [r["risk"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(coverage, risk)
    plt.xlabel("Coverage")
    plt.ylabel("Risk (error rate)")
    plt.title("Coverage vs Risk")
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def infer_threshold_from_validation(
    model: BayesianMLP,
    guide,
    preprocessor,
    feature_names: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_mc_samples: int,
) -> float:
    """
    Tune threshold on validation data using maximum F1.
    """
    X_val_transformed = preprocessor.transform(X_val[feature_names]).astype(np.float32)
    X_val_tensor = torch.tensor(X_val_transformed, dtype=torch.float32, device=DEVICE)

    y_val_proba = predict_with_uncertainty(
        model=model,
        guide=guide,
        x=X_val_tensor,
        num_samples=num_mc_samples,
    )[0]

    threshold, _ = find_best_f1_threshold(
        y_true=y_val.to_numpy(dtype=np.int64),
        y_proba=y_val_proba,
    )
    return float(threshold)


def run_uncertainty_analysis(
    data_path: str | Path = DATA_PATH,
    checkpoint_path: str | Path = BNN_CHECKPOINT_PATH,
    preprocessor_path: str | Path = BNN_PREPROCESSOR_PATH,
    output_dir: str | Path = BNN_UNCERTAINTY_DIR,
    split: str = "test",
    threshold: float | None = None,
    num_mc_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
) -> dict[str, Any]:
    """
    Run uncertainty analysis for the trained BNN.

    Logic:
    - If threshold is provided, use it directly.
    - If threshold is None and split='test', tune threshold on validation and apply to test.
    - If threshold is None and split='validation', tune threshold on validation itself.
    """
    output_dir = ensure_dir(output_dir)

    df = load_data(data_path, add_row_id=True)
    splits = split_features_target(df, drop_row_id_from_X=False)

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")

    if split == "validation":
        X_eval = splits.X_val.copy()
        y_eval = splits.y_val
    else:
        X_eval = splits.X_test.copy()
        y_eval = splits.y_test

    if ROW_ID_COLUMN not in X_eval.columns:
        raise ValueError(f"Expected '{ROW_ID_COLUMN}' column in X_eval but it was not found.")

    row_ids = X_eval[ROW_ID_COLUMN].to_numpy()

    model, guide, preprocessor, feature_names = load_bnn_artifacts(
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
    )

    if feature_names is None:
        feature_names = get_feature_columns(
            X_eval.columns,
            exclude_target=False,
            exclude_row_id=True,
        )

    # Tune threshold on validation if not provided
    if threshold is None:
        threshold = infer_threshold_from_validation(
            model=model,
            guide=guide,
            preprocessor=preprocessor,
            feature_names=feature_names,
            X_val=splits.X_val.copy(),
            y_val=splits.y_val,
            num_mc_samples=num_mc_samples,
        )

    X_transformed = preprocessor.transform(X_eval[feature_names]).astype(np.float32)
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32, device=DEVICE)

    mean_probs, std_probs = predict_with_uncertainty(
        model=model,
        guide=guide,
        x=X_tensor,
        num_samples=num_mc_samples,
    )

    y_true = y_eval.to_numpy(dtype=np.int64)
    y_pred = (mean_probs >= threshold).astype(int)
    confusion_type = assign_confusion_label(y_true, y_pred)

    results_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "predicted_probability": mean_probs,
            "uncertainty_std": std_probs,
            "confusion_type": confusion_type,
        }
    )

    summary = summarize_uncertainty(results_df)

    coverage_risk_results = compute_coverage_risk(results_df)

    max_uncertainty = float(results_df["uncertainty_std"].max())
    selective_results: dict[str, Any] = {}
    if max_uncertainty == 0.0:
        threshold_grid = [0.0]
    else:
        threshold_grid = np.linspace(0.0, max_uncertainty, 20)

    for thr in threshold_grid:
        selective_results[f"{float(thr):.8f}"] = selective_metrics(results_df, float(thr))

    summary["metadata"] = {
        "split": split,
        "threshold": float(threshold),
        "num_mc_samples": int(num_mc_samples),
        "device": DEVICE,
        "checkpoint_path": str(checkpoint_path),
        "preprocessor_path": str(preprocessor_path),
        "row_id_included": True,
        "threshold_selection": "validation_optimized_f1" if threshold is not None else "fixed_user_threshold",
    }
    summary["coverage_risk"] = coverage_risk_results
    summary["selective_prediction"] = selective_results

    csv_path = output_dir / f"{split}_uncertainty_per_sample.csv"
    json_path = output_dir / f"{split}_uncertainty_summary.json"
    hist_path = output_dir / f"{split}_uncertainty_by_true_class.png"
    boxplot_path = output_dir / f"{split}_uncertainty_by_confusion_type.png"
    scatter_path = output_dir / f"{split}_probability_vs_uncertainty.png"
    coverage_risk_json_path = output_dir / f"{split}_coverage_risk.json"
    coverage_risk_plot_path = output_dir / f"{split}_coverage_risk.png"
    selective_json_path = output_dir / f"{split}_selective_prediction.json"

    results_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with coverage_risk_json_path.open("w", encoding="utf-8") as f:
        json.dump(coverage_risk_results, f, indent=2)

    with selective_json_path.open("w", encoding="utf-8") as f:
        json.dump(selective_results, f, indent=2)

    plot_uncertainty_by_true_class(results_df, hist_path)
    plot_uncertainty_by_confusion_type(results_df, boxplot_path)
    plot_probability_vs_uncertainty(results_df, scatter_path)
    plot_coverage_risk(coverage_risk_results, coverage_risk_plot_path)

    print(f"Saved per-sample uncertainty CSV to: {csv_path}")
    print(f"Saved uncertainty summary JSON to: {json_path}")
    print(f"Saved histogram to: {hist_path}")
    print(f"Saved boxplot to: {boxplot_path}")
    print(f"Saved scatter plot to: {scatter_path}")
    print(f"Saved coverage-risk JSON to: {coverage_risk_json_path}")
    print(f"Saved coverage-risk plot to: {coverage_risk_plot_path}")
    print(f"Saved selective prediction JSON to: {selective_json_path}")

    return summary


if __name__ == "__main__":
    run_uncertainty_analysis()