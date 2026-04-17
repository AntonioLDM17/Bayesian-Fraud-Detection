from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import EXPERIMENTS_DIR, FULL_EVALUATION_PATH, PLOT_DPI


CALIBRATION_RESULTS_DIR = EXPERIMENTS_DIR / "calibration_results"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_model_name(model_name: str) -> str:
    """
    Convert qualified model name like 'baseline_results/logistic_regression'
    into a filesystem-friendly name.
    """
    return model_name.replace("/", "__")


def plot_reliability_diagram(
    calibration_table: list[dict[str, Any]],
    model_name: str,
    split_name: str,
    output_path: str | Path,
) -> None:
    """
    Plot a reliability diagram from a calibration table.
    """
    if not calibration_table:
        return

    df = pd.DataFrame(calibration_table)
    if df.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    plt.plot(
        df["mean_confidence"],
        df["empirical_accuracy"],
        marker="o",
        linewidth=1.5,
        label="Model",
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability Diagram — {model_name} ({split_name})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def plot_confidence_histogram(
    calibration_table: list[dict[str, Any]],
    model_name: str,
    split_name: str,
    output_path: str | Path,
) -> None:
    """
    Plot a simple confidence histogram using calibration bins.
    """
    if not calibration_table:
        return

    df = pd.DataFrame(calibration_table)
    if df.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = df["bin_center"]
    heights = df["count"]
    widths = df["bin_right"] - df["bin_left"]

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(x, heights, width=widths * 0.95, align="center")
    plt.xlabel("Predicted probability bin")
    plt.ylabel("Count")
    plt.title(f"Confidence Histogram — {model_name} ({split_name})")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI)
    plt.close()


def summarize_calibration(eval_data: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compact calibration summary from full evaluation results.
    """
    summary: dict[str, Any] = {"models": {}}

    models = eval_data.get("models", {})
    for model_name, model_info in models.items():
        model_summary: dict[str, Any] = {}
        for split_name in ["validation", "test"]:
            split_info = model_info.get(split_name, {})
            cal_metrics = split_info.get("calibration_metrics", {})
            model_summary[split_name] = {
                "ece": cal_metrics.get("ece"),
                "mce": cal_metrics.get("mce"),
            }
        summary["models"][model_name] = model_summary

    return summary


def run_calibration_plots(
    evaluation_path: str | Path = FULL_EVALUATION_PATH,
    output_dir: str | Path = CALIBRATION_RESULTS_DIR,
) -> dict[str, Any]:
    """
    Generate calibration plots for all models and both splits
    from experiments/full_evaluation.json.
    """
    evaluation_path = Path(evaluation_path)
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Could not find evaluation file: {evaluation_path}")

    output_dir = ensure_dir(output_dir)

    with evaluation_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    models = eval_data.get("models", {})
    if not models:
        raise ValueError("No models found in evaluation file.")

    for model_name, model_info in models.items():
        safe_name = normalize_model_name(model_name)

        for split_name in ["validation", "test"]:
            split_info = model_info.get(split_name, {})
            calibration_table = split_info.get("calibration_table", [])

            if not calibration_table:
                continue

            reliability_path = output_dir / f"{safe_name}_{split_name}_reliability.png"
            histogram_path = output_dir / f"{safe_name}_{split_name}_confidence_histogram.png"

            plot_reliability_diagram(
                calibration_table=calibration_table,
                model_name=model_name,
                split_name=split_name,
                output_path=reliability_path,
            )
            plot_confidence_histogram(
                calibration_table=calibration_table,
                model_name=model_name,
                split_name=split_name,
                output_path=histogram_path,
            )

            print(f"Saved reliability diagram to: {reliability_path}")
            print(f"Saved confidence histogram to: {histogram_path}")

    summary = summarize_calibration(eval_data)
    summary["metadata"] = {
        "evaluation_path": str(evaluation_path),
        "output_dir": str(output_dir),
    }

    summary_path = output_dir / "calibration_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved calibration summary to: {summary_path}")
    return summary


if __name__ == "__main__":
    run_calibration_plots()