from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.gplvm import GPLVM


RANDOM_STATE = 42
TARGET_COLUMN = "Class"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: str | Path) -> pd.DataFrame:
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


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def sample_latent_dataset(
    df: pd.DataFrame,
    nonfraud_multiplier: int = 3,
    max_nonfraud: int | None = 1500,
) -> pd.DataFrame:
    """
    Build a manageable subset for GPLVM:
    - all frauds
    - a random sample of non-frauds

    Since GPLVM scales cubically with N, keep this subset moderate.
    """
    fraud_df = df[df[TARGET_COLUMN] == 1].copy()
    nonfraud_df = df[df[TARGET_COLUMN] == 0].copy()

    n_fraud = len(fraud_df)
    n_nonfraud_target = n_fraud * nonfraud_multiplier

    if max_nonfraud is not None:
        n_nonfraud_target = min(n_nonfraud_target, max_nonfraud)

    n_nonfraud_target = min(n_nonfraud_target, len(nonfraud_df))

    sampled_nonfraud = nonfraud_df.sample(
        n=n_nonfraud_target,
        random_state=RANDOM_STATE,
        replace=False,
    )

    subset = pd.concat([fraud_df, sampled_nonfraud], axis=0)
    subset = subset.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return subset


def make_pca_initialization(Y: np.ndarray, latent_dim: int = 2) -> np.ndarray:
    """
    PCA initialization for latent coordinates.
    """
    pca = PCA(n_components=latent_dim, random_state=RANDOM_STATE)
    return pca.fit_transform(Y)


def plot_latent_space(
    latent_df: pd.DataFrame,
    output_path: Path,
    color_column: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 6))

    unique_values = sorted(latent_df[color_column].unique())
    for value in unique_values:
        sub = latent_df[latent_df[color_column] == value]
        plt.scatter(
            sub["z1"],
            sub["z2"],
            s=18,
            alpha=0.65,
            label=f"{color_column} = {value}",
        )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_latent_by_amount(
    latent_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latent_df["z1"],
        latent_df["z2"],
        c=latent_df["Amount"],
        s=18,
        alpha=0.75,
    )
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("GPLVM latent space colored by transaction amount")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Amount")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def train_gplvm(
    Y: np.ndarray,
    latent_dim: int = 2,
    num_epochs: int = 1500,
    learning_rate: float = 0.03,
    latent_reg_weight: float = 1e-3,
    print_every: int = 50,
) -> tuple[GPLVM, dict[str, Any]]:
    """
    Train deterministic GPLVM by maximizing marginal likelihood.
    """
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=DEVICE)

    X_init = make_pca_initialization(Y, latent_dim=latent_dim)
    X_init_tensor = torch.tensor(X_init, dtype=torch.float32, device=DEVICE)

    model = GPLVM(
        Y=Y_tensor,
        latent_dim=latent_dim,
        X_init=X_init_tensor,
        ard=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {
        "loss": [],
    }

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = model.loss(latent_reg_weight=latent_reg_weight)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        history["loss"].append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_state = {
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            }

        if epoch % print_every == 0 or epoch == 1 or epoch == num_epochs:
            hypers = model.get_hyperparameters()
            print(
                f"Epoch {epoch:04d}/{num_epochs} | "
                f"Loss={loss_value:.4f} | "
                f"Noise={hypers['noise']:.6f} | "
                f"Outputscale={hypers['outputscale']:.4f}"
            )

    if best_state is None:
        raise RuntimeError("GPLVM training failed to store a best state.")

    model.load_state_dict(best_state["model_state_dict"])

    training_summary = {
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "latent_reg_weight": float(latent_reg_weight),
    }

    return model, {"history": history, "training_summary": training_summary}


def train_and_save(
    data_path: str | Path = "data/raw/creditcard.csv",
    output_dir: str | Path = "experiments/gplvm_results",
) -> dict[str, Any]:
    """
    Train GPLVM on a fraud-focused subset and save latent embeddings + plots.
    """
    set_seed()
    output_dir = ensure_dir(output_dir)

    df = load_data(data_path)
    subset_df = sample_latent_dataset(
        df=df,
        nonfraud_multiplier=3,
        max_nonfraud=1500,
    )

    feature_cols = [col for col in subset_df.columns if col != TARGET_COLUMN]
    Y_raw = subset_df[feature_cols].copy()

    preprocessor = build_preprocessor()
    Y = preprocessor.fit_transform(Y_raw).astype(np.float32)

    print(f"Training GPLVM on subset with shape: {Y.shape}")
    print(f"Fraud count: {(subset_df[TARGET_COLUMN] == 1).sum()}")
    print(f"Non-fraud count: {(subset_df[TARGET_COLUMN] == 0).sum()}")

    model, training_info = train_gplvm(
        Y=Y,
        latent_dim=2,
        num_epochs=1500,
        learning_rate=0.03,
        latent_reg_weight=1e-3,
        print_every=50,
    )

    latent_positions = model.get_latent_positions().cpu().numpy()
    latent_df = subset_df.copy()
    latent_df["z1"] = latent_positions[:, 0]
    latent_df["z2"] = latent_positions[:, 1]

    checkpoint_path = output_dir / "gplvm.pt"
    preprocessor_path = output_dir / "preprocessor.joblib"
    latent_csv_path = output_dir / "latent_embeddings.csv"
    summary_json_path = output_dir / "training_summary.json"
    plot_class_path = output_dir / "latent_space_by_class.png"
    plot_amount_path = output_dir / "latent_space_by_amount.png"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_shape": list(Y.shape),
            "latent_dim": 2,
            "training_info": training_info,
            "hyperparameters": model.get_hyperparameters(),
            "feature_columns": feature_cols,
        },
        checkpoint_path,
    )
    joblib.dump(preprocessor, preprocessor_path)
    latent_df.to_csv(latent_csv_path, index=False)

    summary = {
        "subset_size": int(len(subset_df)),
        "fraud_count": int((subset_df[TARGET_COLUMN] == 1).sum()),
        "nonfraud_count": int((subset_df[TARGET_COLUMN] == 0).sum()),
        "training_info": training_info,
        "hyperparameters": model.get_hyperparameters(),
        "checkpoint_path": str(checkpoint_path),
        "preprocessor_path": str(preprocessor_path),
        "latent_csv_path": str(latent_csv_path),
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_latent_space(
        latent_df=latent_df,
        output_path=plot_class_path,
        color_column=TARGET_COLUMN,
        title="GPLVM latent space colored by fraud class",
    )

    plot_latent_by_amount(
        latent_df=latent_df,
        output_path=plot_amount_path,
    )

    print(f"\nSaved GPLVM checkpoint to: {checkpoint_path}")
    print(f"Saved GPLVM preprocessor to: {preprocessor_path}")
    print(f"Saved latent embeddings to: {latent_csv_path}")
    print(f"Saved summary to: {summary_json_path}")
    print(f"Saved class plot to: {plot_class_path}")
    print(f"Saved amount plot to: {plot_amount_path}")

    return summary


if __name__ == "__main__":
    train_and_save()