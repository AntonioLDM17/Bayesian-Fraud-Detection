from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.data.load_data import load_data
from src.data.preprocess import build_standard_preprocessor, get_feature_columns
from src.data.split import split_dataframe, sample_fraud_focused_subset
from src.models.gplvm import GPLVM


RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_pca_initialization(Y: np.ndarray, latent_dim: int = 2) -> np.ndarray:
    """
    PCA initialization for latent coordinates.
    """
    pca = PCA(n_components=latent_dim, random_state=RANDOM_STATE)
    return pca.fit_transform(Y)


def plot_latent_space(
    latent_df,
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
    latent_df,
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
                "model_state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
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
    Train GPLVM on a TEST-split subset and save latent embeddings + plots.
    """
    set_seed()
    output_dir = ensure_dir(output_dir)

    df = load_data(data_path, add_row_id=True)
    frames = split_dataframe(df)
    subset_df = sample_fraud_focused_subset(
        frames.test_df,
        nonfraud_multiplier=3,
        max_nonfraud=1500,
    )

    feature_cols = get_feature_columns(subset_df.columns)
    Y_raw = subset_df[feature_cols].copy()

    preprocessor = build_standard_preprocessor(feature_cols)
    Y = preprocessor.fit_transform(Y_raw).astype(np.float32)

    print(f"Training GPLVM on TEST subset with shape: {Y.shape}")
    print(f"Fraud count: {(subset_df['Class'] == 1).sum()}")
    print(f"Non-fraud count: {(subset_df['Class'] == 0).sum()}")

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
            "row_ids": latent_df["row_id"].tolist(),
            "source_split": "test",
        },
        checkpoint_path,
    )

    joblib.dump(preprocessor, preprocessor_path)
    latent_df.to_csv(latent_csv_path, index=False)

    summary = {
        "subset_size": int(len(subset_df)),
        "fraud_count": int((subset_df["Class"] == 1).sum()),
        "nonfraud_count": int((subset_df["Class"] == 0).sum()),
        "training_info": training_info,
        "hyperparameters": model.get_hyperparameters(),
        "checkpoint_path": str(checkpoint_path),
        "preprocessor_path": str(preprocessor_path),
        "latent_csv_path": str(latent_csv_path),
        "row_id_included": True,
        "source_split": "test",
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_latent_space(
        latent_df=latent_df,
        output_path=plot_class_path,
        color_column="Class",
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