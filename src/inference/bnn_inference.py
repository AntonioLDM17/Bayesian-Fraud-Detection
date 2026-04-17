from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyro
import torch

from src.config import (
    BNN_CHECKPOINT_PATH,
    BNN_PREPROCESSOR_PATH,
    DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
    DEVICE,
)
from src.data.preprocess import get_feature_columns
from src.models.bnn import BayesianMLP
from src.analysis.uncertainty import predict_with_uncertainty


def _build_bnn_from_checkpoint(checkpoint: dict[str, Any]) -> BayesianMLP:
    """
    Rebuild BayesianMLP from checkpoint metadata.

    Dropout is forced to 0.0 so predictive uncertainty reflects only
    the Bayesian posterior over weights.
    """
    input_dim = checkpoint["input_dim"]
    prior_scale = checkpoint["prior_scale"]

    if "hidden_dim_1" in checkpoint and "hidden_dim_2" in checkpoint:
        return BayesianMLP(
            input_dim=input_dim,
            hidden_dim_1=checkpoint["hidden_dim_1"],
            hidden_dim_2=checkpoint["hidden_dim_2"],
            prior_scale=prior_scale,
            dropout_rate=0.0,
        )

    hidden_dim = checkpoint["hidden_dim"]
    return BayesianMLP(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim,
        hidden_dim_2=max(hidden_dim // 2, 16),
        prior_scale=prior_scale,
        dropout_rate=0.0,
    )


def load_bnn_artifacts_for_inference(
    checkpoint_path: str | Path = BNN_CHECKPOINT_PATH,
    preprocessor_path: str | Path = BNN_PREPROCESSOR_PATH,
) -> dict[str, Any]:
    """
    Load everything needed for BNN inference.

    Returns a dict with:
    - model
    - guide
    - preprocessor
    - checkpoint
    - feature_names
    """
    checkpoint_path = Path(checkpoint_path)
    preprocessor_path = Path(preprocessor_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BNN checkpoint not found: {checkpoint_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"BNN preprocessor not found: {preprocessor_path}")

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
    if feature_names is None:
        raise ValueError(
            "Checkpoint does not contain feature_names. "
            "Please retrain/save the BNN with feature_names included."
        )

    return {
        "model": model,
        "guide": guide,
        "preprocessor": preprocessor,
        "checkpoint": checkpoint,
        "feature_names": list(feature_names),
    }


def ensure_dataframe(
    X: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert input into a pandas DataFrame.

    Supported inputs:
    - DataFrame
    - single dict
    - list of dicts
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    if isinstance(X, dict):
        return pd.DataFrame([X])

    if isinstance(X, list):
        if len(X) == 0:
            raise ValueError("Input list is empty.")
        return pd.DataFrame(X)

    raise TypeError(
        "X must be a pandas DataFrame, a dict, or a list of dicts."
    )


def validate_and_align_features(
    X: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Ensure the input DataFrame contains all required model features
    and return them in the exact training order.
    """
    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns for BNN inference: "
            f"{missing}"
        )

    return X[feature_names].copy()


def transform_features(
    X: pd.DataFrame,
    preprocessor,
    feature_names: list[str],
) -> np.ndarray:
    """
    Apply the fitted preprocessor to the provided features.
    """
    X_aligned = validate_and_align_features(X, feature_names)
    return preprocessor.transform(X_aligned).astype(np.float32)


@torch.no_grad()
def predict_proba_and_uncertainty(
    model: BayesianMLP,
    guide,
    X_transformed: np.ndarray,
    num_mc_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute predictive mean probability and predictive uncertainty std.
    """
    X_tensor = torch.tensor(X_transformed, dtype=torch.float32, device=DEVICE)

    mean_probs, std_probs = predict_with_uncertainty(
        model=model,
        guide=guide,
        x=X_tensor,
        num_samples=num_mc_samples,
    )
    return mean_probs, std_probs


def predict_batch(
    X: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    checkpoint_path: str | Path = BNN_CHECKPOINT_PATH,
    preprocessor_path: str | Path = BNN_PREPROCESSOR_PATH,
    num_mc_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
    include_input_columns: bool = False,
) -> pd.DataFrame:
    """
    Run BNN inference on one or many samples.

    Returns a DataFrame with:
    - predicted_probability
    - uncertainty_std

    If include_input_columns=True, original input columns are preserved.
    """
    artifacts = load_bnn_artifacts_for_inference(
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
    )

    X_df = ensure_dataframe(X)
    X_transformed = transform_features(
        X=X_df,
        preprocessor=artifacts["preprocessor"],
        feature_names=artifacts["feature_names"],
    )

    mean_probs, std_probs = predict_proba_and_uncertainty(
        model=artifacts["model"],
        guide=artifacts["guide"],
        X_transformed=X_transformed,
        num_mc_samples=num_mc_samples,
    )

    results = pd.DataFrame(
        {
            "predicted_probability": mean_probs,
            "uncertainty_std": std_probs,
        }
    )

    if include_input_columns:
        return pd.concat(
            [X_df.reset_index(drop=True), results.reset_index(drop=True)],
            axis=1,
        )

    return results


def predict_single(
    x: dict[str, Any] | pd.Series,
    checkpoint_path: str | Path = BNN_CHECKPOINT_PATH,
    preprocessor_path: str | Path = BNN_PREPROCESSOR_PATH,
    num_mc_samples: int = DEFAULT_BNN_UNCERTAINTY_MC_SAMPLES,
) -> dict[str, float]:
    """
    Run BNN inference for a single transaction.

    Returns:
    - predicted_probability
    - uncertainty_std
    """
    if isinstance(x, pd.Series):
        x = x.to_dict()

    results = predict_batch(
        X=x,
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
        num_mc_samples=num_mc_samples,
        include_input_columns=False,
    )

    row = results.iloc[0]
    return {
        "predicted_probability": float(row["predicted_probability"]),
        "uncertainty_std": float(row["uncertainty_std"]),
    }