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
from sklearn.model_selection import train_test_split

from src.models.bnn import BayesianMLP


RANDOM_STATE = 42
TARGET_COLUMN = "Class"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Must match the split logic used in training/evaluation.
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
) -> tuple[BayesianMLP, Any]:
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

    return model, guide, preprocessor


@torch.no_grad()
def predict_with_uncertainty(
    model: BayesianMLP,
    guide,
    x: torch.Tensor,
    num_samples: int = 200,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
    - predictive mean probability
    - predictive std of probability across MC samples

    The std is a simple uncertainty proxy.
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
        logits_samples = samples["_RETURN"]  # [num_samples, batch]
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
    plt.savefig(output_path, dpi=200)
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
    plt.savefig(output_path, dpi=200)
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
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_uncertainty_analysis(
    data_path: str | Path = "data/raw/creditcard.csv",
    checkpoint_path: str | Path = "experiments/bnn_results/bayesian_neural_network.pt",
    preprocessor_path: str | Path = "experiments/bnn_results/preprocessor.joblib",
    output_dir: str | Path = "experiments/bnn_results/uncertainty_analysis",
    split: str = "test",
    threshold: float = 0.5,
    num_mc_samples: int = 200,
) -> dict[str, Any]:
    """
    Run uncertainty analysis for the trained BNN.

    Args:
        split: 'validation' or 'test'
    """
    output_dir = ensure_dir(output_dir)

    df = load_data(data_path)
    _, X_val, X_test, _, y_val, y_test = split_data(df)

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")

    if split == "validation":
        X_eval = X_val
        y_eval = y_val
    else:
        X_eval = X_test
        y_eval = y_test

    model, guide, preprocessor = load_bnn_artifacts(
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
    )

    X_transformed = preprocessor.transform(X_eval).astype(np.float32)
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
            "y_true": y_true,
            "y_pred": y_pred,
            "predicted_probability": mean_probs,
            "uncertainty_std": std_probs,
            "confusion_type": confusion_type,
        }
    )

    summary = summarize_uncertainty(results_df)

    summary["metadata"] = {
        "split": split,
        "threshold": float(threshold),
        "num_mc_samples": int(num_mc_samples),
        "device": DEVICE,
        "checkpoint_path": str(checkpoint_path),
        "preprocessor_path": str(preprocessor_path),
    }

    csv_path = output_dir / f"{split}_uncertainty_per_sample.csv"
    json_path = output_dir / f"{split}_uncertainty_summary.json"
    hist_path = output_dir / f"{split}_uncertainty_by_true_class.png"
    boxplot_path = output_dir / f"{split}_uncertainty_by_confusion_type.png"
    scatter_path = output_dir / f"{split}_probability_vs_uncertainty.png"

    results_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_uncertainty_by_true_class(results_df, hist_path)
    plot_uncertainty_by_confusion_type(results_df, boxplot_path)
    plot_probability_vs_uncertainty(results_df, scatter_path)

    print(f"Saved per-sample uncertainty CSV to: {csv_path}")
    print(f"Saved uncertainty summary JSON to: {json_path}")
    print(f"Saved histogram to: {hist_path}")
    print(f"Saved boxplot to: {boxplot_path}")
    print(f"Saved scatter plot to: {scatter_path}")

    return summary


if __name__ == "__main__":
    run_uncertainty_analysis()