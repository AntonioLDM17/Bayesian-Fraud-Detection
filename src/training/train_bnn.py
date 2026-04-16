from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.config import (
    BNN_BATCH_SIZE,
    BNN_CHECKPOINT_PATH,
    BNN_DROPOUT_RATE,
    BNN_EARLY_STOPPING_PATIENCE,
    BNN_EVAL_MC_SAMPLES,
    BNN_HIDDEN_DIM_1,
    BNN_HIDDEN_DIM_2,
    BNN_LEARNING_RATE,
    BNN_METRICS_PATH,
    BNN_MIN_DELTA,
    BNN_NUM_EPOCHS,
    BNN_PREPROCESSOR_PATH,
    BNN_PRIOR_SCALE,
    BNN_RESULTS_DIR,
    BNN_TRAIN_MC_SAMPLES,
    DATA_PATH,
    DEFAULT_THRESHOLD,
    DEVICE,
    RANDOM_STATE,
)
from src.data.load_data import load_data
from src.data.preprocess import build_standard_preprocessor, get_feature_columns
from src.data.split import split_features_target
from src.evaluation.thresholds import find_best_f1_threshold
from src.models.bnn import BayesianMLP, predict_proba_mc
from src.utils.seed import set_seed


def make_weighted_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
) -> DataLoader:
    """
    Create a weighted dataloader to oversample the minority class.
    """
    class_counts = np.bincount(y.astype(int))
    if len(class_counts) < 2 or class_counts[1] == 0:
        raise ValueError("Training set must contain both classes.")

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def evaluate_probabilities(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evaluate probability predictions at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_bnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim_1: int = BNN_HIDDEN_DIM_1,
    hidden_dim_2: int = BNN_HIDDEN_DIM_2,
    prior_scale: float = BNN_PRIOR_SCALE,
    dropout_rate: float = BNN_DROPOUT_RATE,
    learning_rate: float = BNN_LEARNING_RATE,
    num_epochs: int = BNN_NUM_EPOCHS,
    batch_size: int = BNN_BATCH_SIZE,
    num_mc_samples: int = BNN_TRAIN_MC_SAMPLES,
    early_stopping_patience: int = BNN_EARLY_STOPPING_PATIENCE,
    min_delta: float = BNN_MIN_DELTA,
) -> tuple[BayesianMLP, AutoDiagonalNormal, dict[str, Any]]:
    """
    Train Bayesian Neural Network with Pyro SVI using early stopping on validation PR-AUC.
    """
    pyro.clear_param_store()

    input_dim = X_train.shape[1]
    model = BayesianMLP(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        prior_scale=prior_scale,
        dropout_rate=0.0,  # forced off for interpretability
    ).to(DEVICE)

    blocked_model = pyro.poutine.block(model, hide=["obs"])
    guide = AutoDiagonalNormal(blocked_model)

    optimizer = Adam({"lr": learning_rate})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    train_loader = make_weighted_dataloader(X_train, y_train, batch_size=batch_size)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
    }

    best_val_pr_auc = -np.inf
    best_epoch = 0
    patience_counter = 0

    best_model_state = None
    best_param_store_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            loss = svi.step(batch_x, batch_y)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(X_train)
        history["train_loss"].append(float(avg_epoch_loss))

        val_proba = predict_proba_mc(
            model=model,
            guide=guide,
            x=X_val_tensor,
            num_samples=num_mc_samples,
            device=DEVICE,
        )
        val_metrics = evaluate_probabilities(y_val, val_proba, threshold=DEFAULT_THRESHOLD)

        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Loss={avg_epoch_loss:.4f} | "
            f"Val PR-AUC={val_metrics['pr_auc']:.4f} | "
            f"Val F1={val_metrics['f1']:.4f} | "
            f"Val Recall={val_metrics['recall']:.4f}"
        )

        improved = val_metrics["pr_auc"] > (best_val_pr_auc + min_delta)
        if improved:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            patience_counter = 0

            best_model_state = copy.deepcopy(model.state_dict())
            best_param_store_state = copy.deepcopy(pyro.get_param_store().get_state())
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with Val PR-AUC={best_val_pr_auc:.4f}"
            )
            break

    if best_model_state is None or best_param_store_state is None:
        raise RuntimeError("Training finished without storing a best model state.")

    model.load_state_dict(best_model_state)
    pyro.clear_param_store()
    pyro.get_param_store().set_state(best_param_store_state)

    training_summary = {
        "best_epoch": int(best_epoch),
        "best_val_pr_auc": float(best_val_pr_auc),
        "stopped_early": bool(patience_counter >= early_stopping_patience),
        "num_epochs_completed": len(history["train_loss"]),
    }

    return model, guide, {"history": history, "training_summary": training_summary}


def train_and_save(
    data_path: str | Path,
    output_dir: str | Path = BNN_RESULTS_DIR,
) -> dict[str, Any]:
    """
    Train BNN, evaluate it, and save the best checkpoint + preprocessing.

    Final reporting logic:
    - tune threshold on validation
    - reuse the same threshold on test
    """
    set_seed(RANDOM_STATE)
    output_dir = ensure_dir(output_dir)

    df = load_data(data_path, add_row_id=True)
    splits = split_features_target(df, drop_row_id_from_X=False)

    X_train_df = splits.X_train
    X_val_df = splits.X_val
    X_test_df = splits.X_test
    y_train = splits.y_train
    y_val = splits.y_val
    y_test = splits.y_test

    feature_names = get_feature_columns(
        X_train_df.columns,
        exclude_target=False,
        exclude_row_id=True,
    )
    preprocessor = build_standard_preprocessor(feature_names)

    X_train = preprocessor.fit_transform(X_train_df[feature_names]).astype(np.float32)
    X_val = preprocessor.transform(X_val_df[feature_names]).astype(np.float32)
    X_test = preprocessor.transform(X_test_df[feature_names]).astype(np.float32)

    y_train_np = y_train.to_numpy(dtype=np.int64)
    y_val_np = y_val.to_numpy(dtype=np.int64)
    y_test_np = y_test.to_numpy(dtype=np.int64)

    hidden_dim_1 = BNN_HIDDEN_DIM_1
    hidden_dim_2 = BNN_HIDDEN_DIM_2
    prior_scale = BNN_PRIOR_SCALE
    dropout_rate = 0.0
    learning_rate = BNN_LEARNING_RATE
    num_epochs = BNN_NUM_EPOCHS
    batch_size = BNN_BATCH_SIZE
    train_mc_samples = BNN_TRAIN_MC_SAMPLES
    eval_mc_samples = BNN_EVAL_MC_SAMPLES
    early_stopping_patience = BNN_EARLY_STOPPING_PATIENCE
    min_delta = BNN_MIN_DELTA

    print("\nTraining bayesian_neural_network...")
    model, guide, training_info = train_bnn(
        X_train=X_train,
        y_train=y_train_np,
        X_val=X_val,
        y_val=y_val_np,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        prior_scale=prior_scale,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_mc_samples=train_mc_samples,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
    )

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    val_proba = predict_proba_mc(
        model=model,
        guide=guide,
        x=X_val_tensor,
        num_samples=eval_mc_samples,
        device=DEVICE,
    )
    test_proba = predict_proba_mc(
        model=model,
        guide=guide,
        x=X_test_tensor,
        num_samples=eval_mc_samples,
        device=DEVICE,
    )

    optimal_threshold, best_val_f1 = find_best_f1_threshold(
        y_true=y_val_np,
        y_proba=val_proba,
    )

    val_metrics = evaluate_probabilities(
        y_true=y_val_np,
        y_proba=val_proba,
        threshold=optimal_threshold,
    )
    test_metrics = evaluate_probabilities(
        y_true=y_test_np,
        y_proba=test_proba,
        threshold=optimal_threshold,
    )

    print(f"Validation-selected optimal threshold: {optimal_threshold:.4f}")
    print(f"Best validation F1 at optimal threshold: {best_val_f1:.4f}")

    print("Validation metrics for bayesian_neural_network:")
    for metric_name, metric_value in val_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("Test metrics for bayesian_neural_network:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    checkpoint_path = output_dir / Path(BNN_CHECKPOINT_PATH).name
    preprocessor_path = output_dir / Path(BNN_PREPROCESSOR_PATH).name
    metrics_path = output_dir / Path(BNN_METRICS_PATH).name

    torch.save(
        {
            "input_dim": X_train.shape[1],
            "hidden_dim_1": hidden_dim_1,
            "hidden_dim_2": hidden_dim_2,
            "prior_scale": prior_scale,
            "dropout_rate": 0.0,
            "model_state_dict": model.state_dict(),
            "pyro_param_store": pyro.get_param_store().get_state(),
            "training_info": training_info,
            "config": {
                "device": DEVICE,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "train_mc_samples": train_mc_samples,
                "eval_mc_samples": eval_mc_samples,
                "early_stopping_patience": early_stopping_patience,
                "min_delta": min_delta,
                "dropout_rate": 0.0,
            },
            "feature_names": feature_names,
            "optimal_threshold": float(optimal_threshold),
            "best_val_f1_at_optimal_threshold": float(best_val_f1),
        },
        checkpoint_path,
    )
    joblib.dump(preprocessor, preprocessor_path)

    results = {
        "bayesian_neural_network": {
            "validation": val_metrics,
            "test": test_metrics,
            "checkpoint_path": str(checkpoint_path),
            "preprocessor_path": str(preprocessor_path),
            "training_info": training_info,
            "optimal_threshold": float(optimal_threshold),
            "best_val_f1_at_optimal_threshold": float(best_val_f1),
        }
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved best BNN checkpoint to: {checkpoint_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")
    print(f"Saved metrics to: {metrics_path}")

    return results


if __name__ == "__main__":
    train_and_save(DATA_PATH)