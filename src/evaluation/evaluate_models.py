from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyro
import torch

from src.config import (
    BNN_PREPROCESSOR_PATH,
    DATA_PATH,
    DEFAULT_BNN_MC_SAMPLES,
    DEFAULT_MODEL_DIRS,
    DEFAULT_N_BINS,
    DEFAULT_THRESHOLD,
    DEVICE,
    FULL_EVALUATION_PATH,
)
from src.data.load_data import load_data
from src.data.preprocess import get_feature_columns
from src.data.split import split_features_target
from src.evaluation.calibration import calibration_metrics, calibration_table
from src.evaluation.metrics import evaluate_binary_classifier
from src.evaluation.proper_scoring import probabilistic_metrics
from src.models.bnn import BayesianMLP, predict_proba_mc


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_probabilities(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Compute full evaluation from probabilities."""
    classification = evaluate_binary_classifier(
        y_true=y_true,
        y_proba=y_proba,
        threshold=threshold,
    )

    scoring = probabilistic_metrics(
        y_true=y_true,
        y_proba=y_proba,
    )

    calibration = calibration_metrics(
        y_true=y_true,
        y_proba=y_proba,
        n_bins=n_bins,
    )

    calib_table = calibration_table(
        y_true=y_true,
        y_proba=y_proba,
        n_bins=n_bins,
    )

    return {
        "classification_metrics": classification,
        "proper_scoring_rules": scoring,
        "calibration_metrics": calibration,
        "calibration_table": calib_table,
    }


def evaluate_sklearn_model(
    model_path: str | Path,
    X,
    y,
    threshold: float = DEFAULT_THRESHOLD,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Load a saved sklearn model and compute full evaluation."""
    model = joblib.load(model_path)
    y_proba = model.predict_proba(X)[:, 1]

    return evaluate_probabilities(
        y_true=np.asarray(y),
        y_proba=y_proba,
        threshold=threshold,
        n_bins=n_bins,
    )


def _build_bnn_from_checkpoint(checkpoint: dict[str, Any]) -> BayesianMLP:
    """
    Build BayesianMLP from checkpoint metadata.

    Supports both:
    - new checkpoints: hidden_dim_1, hidden_dim_2, dropout_rate
    - old checkpoints: hidden_dim
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

    hidden_dim = checkpoint["hidden_dim"]
    return BayesianMLP(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim,
        hidden_dim_2=max(hidden_dim // 2, 16),
        prior_scale=prior_scale,
        dropout_rate=0.0,
    )


def load_bnn_artifacts(
    checkpoint_path: str | Path,
    preprocessor_path: str | Path,
) -> tuple[BayesianMLP, Any, Any]:
    """
    Load BNN checkpoint and corresponding preprocessor.
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


def evaluate_bnn_model(
    checkpoint_path: str | Path,
    preprocessor_path: str | Path,
    X,
    y,
    threshold: float = DEFAULT_THRESHOLD,
    n_bins: int = DEFAULT_N_BINS,
    num_mc_samples: int = DEFAULT_BNN_MC_SAMPLES,
) -> dict[str, Any]:
    """
    Evaluate saved BNN checkpoint using Monte Carlo predictive probabilities.
    """
    model, guide, preprocessor = load_bnn_artifacts(
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    feature_names = checkpoint.get("feature_names")

    if feature_names is None:
        feature_names = get_feature_columns(X.columns, exclude_target=False, exclude_row_id=True)

    X_transformed = preprocessor.transform(X[feature_names]).astype(np.float32)
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32, device=DEVICE)

    y_proba = predict_proba_mc(
        model=model,
        guide=guide,
        x=X_tensor,
        num_samples=num_mc_samples,
        device=DEVICE,
    )

    return evaluate_probabilities(
        y_true=np.asarray(y),
        y_proba=y_proba,
        threshold=threshold,
        n_bins=n_bins,
    )


def collect_model_artifacts(model_dirs: list[str | Path]) -> list[dict[str, Any]]:
    """
    Collect supported model artifacts from one or more experiment directories.

    Supported:
    - sklearn/joblib models (*.joblib), excluding preprocessor.joblib
    - BNN checkpoints (*.pt) that have a sibling preprocessor.joblib
    """
    collected: list[dict[str, Any]] = []

    for model_dir in model_dirs:
        model_dir = Path(model_dir)

        if not model_dir.exists():
            print(f"Warning: model directory does not exist: {model_dir}")
            continue

        group_name = model_dir.name

        for model_path in sorted(model_dir.glob("*.joblib")):
            if model_path.name == "preprocessor.joblib":
                continue

            model_name = model_path.stem
            qualified_name = f"{group_name}/{model_name}"

            collected.append(
                {
                    "qualified_name": qualified_name,
                    "artifact_type": "sklearn",
                    "model_path": model_path,
                }
            )

        for checkpoint_path in sorted(model_dir.glob("*.pt")):
            model_name = checkpoint_path.stem
            qualified_name = f"{group_name}/{model_name}"

            preprocessor_path = model_dir / Path(BNN_PREPROCESSOR_PATH).name
            if not preprocessor_path.exists():
                print(
                    "Warning: found BNN checkpoint but missing preprocessor.joblib in "
                    f"{model_dir}"
                )
                continue

            collected.append(
                {
                    "qualified_name": qualified_name,
                    "artifact_type": "bnn",
                    "model_path": checkpoint_path,
                    "preprocessor_path": preprocessor_path,
                }
            )

    return collected


def evaluate_artifact(
    artifact: dict[str, Any],
    X,
    y,
    threshold: float = DEFAULT_THRESHOLD,
    n_bins: int = DEFAULT_N_BINS,
    bnn_mc_samples: int = DEFAULT_BNN_MC_SAMPLES,
) -> dict[str, Any]:
    """
    Evaluate one collected artifact according to its type.
    """
    artifact_type = artifact["artifact_type"]

    if artifact_type == "sklearn":
        return evaluate_sklearn_model(
            model_path=artifact["model_path"],
            X=X,
            y=y,
            threshold=threshold,
            n_bins=n_bins,
        )

    if artifact_type == "bnn":
        return evaluate_bnn_model(
            checkpoint_path=artifact["model_path"],
            preprocessor_path=artifact["preprocessor_path"],
            X=X,
            y=y,
            threshold=threshold,
            n_bins=n_bins,
            num_mc_samples=bnn_mc_samples,
        )

    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def print_summary(model_name: str, split_name: str, results: dict[str, Any]) -> None:
    """Print compact summary of the most important metrics."""
    cls = results["classification_metrics"]
    prob = results["proper_scoring_rules"]
    cal = results["calibration_metrics"]

    print(
        f"[{split_name}] {model_name} | "
        f"PR-AUC={cls['pr_auc']:.4f} | "
        f"F1={cls['f1']:.4f} | "
        f"Recall={cls['recall']:.4f} | "
        f"NLL={prob['nll']:.4f} | "
        f"Brier={prob['brier_score']:.4f} | "
        f"ECE={cal['ece']:.4f}"
    )


def evaluate_all_models(
    data_path: str | Path = DATA_PATH,
    model_dirs: list[str | Path] | None = None,
    output_path: str | Path = FULL_EVALUATION_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    n_bins: int = DEFAULT_N_BINS,
    bnn_mc_samples: int = DEFAULT_BNN_MC_SAMPLES,
) -> dict[str, Any]:
    """
    Evaluate all saved models from one or more directories on validation and test splits.
    """
    if model_dirs is None:
        model_dirs = DEFAULT_MODEL_DIRS

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    df = load_data(data_path, add_row_id=True)
    splits = split_features_target(df, drop_row_id_from_X=True)

    X_val = splits.X_val
    X_test = splits.X_test
    y_val = splits.y_val
    y_test = splits.y_test

    collected_artifacts = collect_model_artifacts(model_dirs)
    if not collected_artifacts:
        raise FileNotFoundError(
            "No supported model artifacts found in any provided directory."
        )

    results: dict[str, Any] = {
        "metadata": {
            "data_path": str(data_path),
            "threshold": float(threshold),
            "n_bins": int(n_bins),
            "bnn_mc_samples": int(bnn_mc_samples),
            "model_dirs": [str(p) for p in model_dirs],
            "device": DEVICE,
        },
        "models": {},
    }

    for artifact in collected_artifacts:
        qualified_name = artifact["qualified_name"]
        artifact_type = artifact["artifact_type"]

        print(f"Evaluating {qualified_name} ({artifact_type})...")

        val_results = evaluate_artifact(
            artifact=artifact,
            X=X_val,
            y=y_val,
            threshold=threshold,
            n_bins=n_bins,
            bnn_mc_samples=bnn_mc_samples,
        )

        test_results = evaluate_artifact(
            artifact=artifact,
            X=X_test,
            y=y_test,
            threshold=threshold,
            n_bins=n_bins,
            bnn_mc_samples=bnn_mc_samples,
        )

        artifact_record: dict[str, Any] = {
            "artifact_type": artifact_type,
            "validation": val_results,
            "test": test_results,
            "model_path": str(artifact["model_path"]),
        }

        if artifact_type == "bnn":
            artifact_record["preprocessor_path"] = str(artifact["preprocessor_path"])

        results["models"][qualified_name] = artifact_record

        print_summary(qualified_name, "validation", val_results)
        print_summary(qualified_name, "test", test_results)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved full evaluation to: {output_path}")
    return results


if __name__ == "__main__":
    evaluate_all_models()
