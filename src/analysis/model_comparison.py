from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import EXPERIMENTS_DIR, FULL_EVALUATION_PATH


MODEL_COMPARISON_TABLE_PATH = EXPERIMENTS_DIR / "model_comparison_table.csv"
MODEL_COMPARISON_MARKDOWN_PATH = EXPERIMENTS_DIR / "model_comparison_table.md"
EFFICIENCY_SUMMARY_PATH = EXPERIMENTS_DIR / "efficiency_results" / "efficiency_summary.json"
EFFICIENCY_TABLE_PATH = EXPERIMENTS_DIR / "efficiency_results" / "efficiency_table.csv"


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_base_model_comparison_table(eval_data: dict[str, Any]) -> pd.DataFrame:
    if not eval_data or "models" not in eval_data:
        return pd.DataFrame()

    rows = []
    for qualified_name, model_info in eval_data["models"].items():
        validation = model_info["validation"]
        test = model_info["test"]

        val_cls = validation["classification_metrics"]
        val_prob = validation["proper_scoring_rules"]
        val_cal = validation["calibration_metrics"]

        test_cls = test["classification_metrics"]
        test_prob = test["proper_scoring_rules"]
        test_cal = test["calibration_metrics"]

        rows.append(
            {
                "model": qualified_name,
                "artifact_type": model_info.get("artifact_type", ""),
                "selected_threshold_from_validation": model_info.get(
                    "selected_threshold_from_validation",
                    validation.get("optimal_threshold"),
                ),
                "pr_auc_val": val_cls["pr_auc"],
                "roc_auc_val": val_cls["roc_auc"],
                "f1_val": val_cls["f1"],
                "precision_val": val_cls["precision"],
                "recall_val": val_cls["recall"],
                "nll_val": val_prob["nll"],
                "brier_val": val_prob["brier_score"],
                "ece_val": val_cal["ece"],
                "pr_auc_test": test_cls["pr_auc"],
                "roc_auc_test": test_cls["roc_auc"],
                "f1_test": test_cls["f1"],
                "precision_test": test_cls["precision"],
                "recall_test": test_cls["recall"],
                "nll_test": test_prob["nll"],
                "brier_test": test_prob["brier_score"],
                "ece_test": test_cal["ece"],
                "model_path": model_info.get("model_path"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pr_auc_test", ascending=False).reset_index(drop=True)

    return df


def merge_efficiency_if_available(
    comparison_df: pd.DataFrame,
    efficiency_table_path: str | Path = EFFICIENCY_TABLE_PATH,
) -> pd.DataFrame:
    if comparison_df.empty:
        return comparison_df

    efficiency_table_path = Path(efficiency_table_path)
    if not efficiency_table_path.exists():
        return comparison_df

    efficiency_df = pd.read_csv(efficiency_table_path)
    if "model" not in efficiency_df.columns:
        return comparison_df

    columns_to_merge = [
        "model",
        "artifact_size_mb",
        "training_time_seconds",
        "load_time_seconds",
        "inference_time_total_seconds",
        "latency_ms_per_sample",
        "mc_samples_used",
    ]
    available_columns = [c for c in columns_to_merge if c in efficiency_df.columns]

    return comparison_df.merge(
        efficiency_df[available_columns],
        on="model",
        how="left",
    )


def summarize_model_comparison(comparison_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_models": int(len(comparison_df)),
    }

    if comparison_df.empty:
        return summary

    best_pr_auc = comparison_df.sort_values("pr_auc_test", ascending=False).iloc[0]
    best_f1 = comparison_df.sort_values("f1_test", ascending=False).iloc[0]
    best_ece = comparison_df.sort_values("ece_test", ascending=True).iloc[0]
    best_nll = comparison_df.sort_values("nll_test", ascending=True).iloc[0]

    summary["best_pr_auc_test"] = {
        "model": best_pr_auc["model"],
        "value": float(best_pr_auc["pr_auc_test"]),
    }
    summary["best_f1_test"] = {
        "model": best_f1["model"],
        "value": float(best_f1["f1_test"]),
    }
    summary["best_ece_test"] = {
        "model": best_ece["model"],
        "value": float(best_ece["ece_test"]),
    }
    summary["best_nll_test"] = {
        "model": best_nll["model"],
        "value": float(best_nll["nll_test"]),
    }

    if "latency_ms_per_sample" in comparison_df.columns:
        valid_latency = comparison_df.dropna(subset=["latency_ms_per_sample"])
        if not valid_latency.empty:
            fastest = valid_latency.sort_values("latency_ms_per_sample", ascending=True).iloc[0]
            summary["fastest_inference"] = {
                "model": fastest["model"],
                "latency_ms_per_sample": float(fastest["latency_ms_per_sample"]),
            }

    return summary


def save_markdown_table(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if df.empty:
        output_path.write_text("No model comparison data available.\n", encoding="utf-8")
        return

    display_cols = [
        col
        for col in [
            "model",
            "artifact_type",
            "pr_auc_test",
            "roc_auc_test",
            "f1_test",
            "recall_test",
            "nll_test",
            "brier_test",
            "ece_test",
            "latency_ms_per_sample",
            "artifact_size_mb",
            "selected_threshold_from_validation",
        ]
        if col in df.columns
    ]

    markdown_df = df[display_cols].copy()
    output_path.write_text(markdown_df.to_markdown(index=False), encoding="utf-8")


def run_model_comparison(
    evaluation_path: str | Path = FULL_EVALUATION_PATH,
    efficiency_table_path: str | Path = EFFICIENCY_TABLE_PATH,
    output_csv_path: str | Path = MODEL_COMPARISON_TABLE_PATH,
    output_md_path: str | Path = MODEL_COMPARISON_MARKDOWN_PATH,
) -> dict[str, Any]:
    """
    Build a final model comparison table from full_evaluation.json
    and merge efficiency metrics if available.
    """
    eval_data = load_json(evaluation_path)
    if not eval_data:
        raise FileNotFoundError(f"Could not find evaluation file: {evaluation_path}")

    comparison_df = build_base_model_comparison_table(eval_data)
    comparison_df = merge_efficiency_if_available(
        comparison_df,
        efficiency_table_path=efficiency_table_path,
    )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv_path, index=False)

    save_markdown_table(comparison_df, output_md_path)

    summary = summarize_model_comparison(comparison_df)
    summary["metadata"] = {
        "evaluation_path": str(evaluation_path),
        "efficiency_table_path": str(efficiency_table_path),
        "output_csv_path": str(output_csv_path),
        "output_md_path": str(output_md_path),
        "efficiency_merged": Path(efficiency_table_path).exists(),
    }

    summary_path = output_csv_path.with_name("model_comparison_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved model comparison table to: {output_csv_path}")
    print(f"Saved model comparison markdown to: {output_md_path}")
    print(f"Saved model comparison summary to: {summary_path}")

    return summary


if __name__ == "__main__":
    run_model_comparison()