from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


TARGET_COLUMN = "Class"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")
    return pd.read_csv(path)


def infer_join_key(
    latent_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> str | None:
    """
    Try to infer a safe join key shared by both dataframes.
    """
    candidate_keys = [
        "row_id",
        "index",
        "sample_index",
        "original_index",
        "transaction_id",
    ]

    for key in candidate_keys:
        if key in latent_df.columns and key in other_df.columns:
            return key

    return None


def merge_optional(
    latent_df: pd.DataFrame,
    other_df: pd.DataFrame | None,
    name: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge an optional dataframe if a shared key exists.
    """
    messages: list[str] = []

    if other_df is None:
        messages.append(f"{name}: not provided.")
        return latent_df, messages

    join_key = infer_join_key(latent_df, other_df)
    if join_key is None:
        messages.append(
            f"{name}: no shared key found, skipping merge."
        )
        return latent_df, messages

    merged = latent_df.merge(other_df, on=join_key, how="left")
    messages.append(f"{name}: merged using key '{join_key}'.")
    return merged, messages


def plot_latent_by_class(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    unique_values = sorted(df[TARGET_COLUMN].dropna().unique())
    for value in unique_values:
        sub = df[df[TARGET_COLUMN] == value]
        plt.scatter(
            sub["z1"],
            sub["z2"],
            s=18,
            alpha=0.65,
            label=f"Class = {value}",
        )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("GPLVM latent space colored by fraud class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_latent_by_amount(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if "Amount" not in df.columns:
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["z1"],
        df["z2"],
        c=df["Amount"],
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


def plot_latent_by_uncertainty(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if "uncertainty_std" not in df.columns:
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["z1"],
        df["z2"],
        c=df["uncertainty_std"],
        s=18,
        alpha=0.75,
    )
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("GPLVM latent space colored by BNN uncertainty")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Predictive uncertainty (std)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_latent_by_confusion_type(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if "confusion_type" not in df.columns:
        return

    plt.figure(figsize=(8, 6))
    labels = ["TN", "TP", "FP", "FN"]

    for label in labels:
        sub = df[df["confusion_type"] == label]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["z1"],
            sub["z2"],
            s=18,
            alpha=0.65,
            label=label,
        )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("GPLVM latent space colored by BNN confusion type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_latent_by_decision(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    if "decision" not in df.columns:
        return

    plt.figure(figsize=(8, 6))
    labels = ["ACCEPT", "REVIEW", "BLOCK"]

    for label in labels:
        sub = df[df["decision"] == label]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["z1"],
            sub["z2"],
            s=18,
            alpha=0.65,
            label=label,
        )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("GPLVM latent space colored by uncertainty-aware decision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def summarize_latent_space(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_points": int(len(df)),
        "columns": list(df.columns),
    }

    if TARGET_COLUMN in df.columns:
        summary["class_counts"] = {
            str(k): int(v)
            for k, v in df[TARGET_COLUMN].value_counts(dropna=False).to_dict().items()
        }

    if "confusion_type" in df.columns:
        summary["confusion_counts"] = {
            str(k): int(v)
            for k, v in df["confusion_type"].value_counts(dropna=False).to_dict().items()
        }

    if "decision" in df.columns:
        summary["decision_counts"] = {
            str(k): int(v)
            for k, v in df["decision"].value_counts(dropna=False).to_dict().items()
        }

    if "uncertainty_std" in df.columns:
        summary["uncertainty_summary"] = {
            "mean_uncertainty": float(df["uncertainty_std"].mean()),
            "median_uncertainty": float(df["uncertainty_std"].median()),
        }

        if TARGET_COLUMN in df.columns:
            summary["uncertainty_by_class"] = {}
            for class_value in sorted(df[TARGET_COLUMN].dropna().unique()):
                sub = df[df[TARGET_COLUMN] == class_value]
                summary["uncertainty_by_class"][str(class_value)] = {
                    "count": int(len(sub)),
                    "mean_uncertainty": float(sub["uncertainty_std"].mean()),
                    "median_uncertainty": float(sub["uncertainty_std"].median()),
                }

    return summary


def run_latent_analysis(
    latent_csv_path: str | Path = "experiments/gplvm_results/latent_embeddings.csv",
    uncertainty_csv_path: str | Path | None = None,
    decision_csv_path: str | Path | None = None,
    output_dir: str | Path = "experiments/gplvm_results/latent_analysis",
) -> dict[str, Any]:
    """
    Analyze GPLVM latent embeddings and optionally enrich them with
    BNN uncertainty and decision outputs.

    Notes:
    - Best results happen when latent_embeddings.csv includes a join key
      also present in uncertainty/decision files.
    """
    output_dir = ensure_dir(output_dir)

    latent_df = load_csv(latent_csv_path)

    uncertainty_df = None
    decision_df = None

    if uncertainty_csv_path is not None and Path(uncertainty_csv_path).exists():
        uncertainty_df = load_csv(uncertainty_csv_path)

    if decision_csv_path is not None and Path(decision_csv_path).exists():
        decision_df = load_csv(decision_csv_path)

    notes: list[str] = []

    merged_df, msgs = merge_optional(latent_df, uncertainty_df, "uncertainty")
    notes.extend(msgs)

    merged_df, msgs = merge_optional(merged_df, decision_df, "decision")
    notes.extend(msgs)

    class_plot = output_dir / "latent_space_by_class.png"
    amount_plot = output_dir / "latent_space_by_amount.png"
    uncertainty_plot = output_dir / "latent_space_by_uncertainty.png"
    confusion_plot = output_dir / "latent_space_by_confusion_type.png"
    decision_plot = output_dir / "latent_space_by_decision.png"
    merged_csv = output_dir / "latent_analysis_merged.csv"
    summary_json = output_dir / "latent_analysis_summary.json"

    plot_latent_by_class(merged_df, class_plot)
    plot_latent_by_amount(merged_df, amount_plot)
    plot_latent_by_uncertainty(merged_df, uncertainty_plot)
    plot_latent_by_confusion_type(merged_df, confusion_plot)
    plot_latent_by_decision(merged_df, decision_plot)

    merged_df.to_csv(merged_csv, index=False)

    summary = summarize_latent_space(merged_df)
    summary["notes"] = notes
    summary["paths"] = {
        "merged_csv": str(merged_csv),
        "class_plot": str(class_plot),
        "amount_plot": str(amount_plot),
        "uncertainty_plot": str(uncertainty_plot),
        "confusion_plot": str(confusion_plot),
        "decision_plot": str(decision_plot),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved merged latent analysis CSV to: {merged_csv}")
    print(f"Saved summary JSON to: {summary_json}")
    print(f"Saved class plot to: {class_plot}")
    print(f"Saved amount plot to: {amount_plot}")

    if "uncertainty_std" in merged_df.columns:
        print(f"Saved uncertainty plot to: {uncertainty_plot}")
    if "confusion_type" in merged_df.columns:
        print(f"Saved confusion plot to: {confusion_plot}")
    if "decision" in merged_df.columns:
        print(f"Saved decision plot to: {decision_plot}")

    return summary


if __name__ == "__main__":
    run_latent_analysis(
        latent_csv_path="experiments/gplvm_results/latent_embeddings.csv",
        uncertainty_csv_path=None,
        decision_csv_path=None,
    )